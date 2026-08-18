# SmartDoc

## Overview

SmartDoc is a RAG-based document Q&A assistant. Employees ask plain-English
questions and get accurate, cited answers drawn from a library of company PDFs
(HR policies, product manuals, onboarding guides, SOPs).

The system answers **only** from retrieved document context. When the context
does not contain the answer, it replies "I don't know based on the available
documents" rather than guessing. Every answer returns structured sources
(document, page, snippet).

**Stack**

| Layer | Choice |
| --- | --- |
| Backend / all RAG logic | FastAPI (Python) |
| Vector DB | ChromaDB (`PersistentClient`, on disk) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini` (low temperature) |
| Relational store | SQLite users, documents, sessions, messages (V2) |
| Auth | FastAPI-owned: bcrypt + JWT, Google OAuth via authlib (V2) |
| Frontend | Next.js App Router + TypeScript + Tailwind + shadcn/ui (V2) |
| Orchestration | LangChain where it helps |

**Layout**

```
smartdoc/
  backend/          # FastAPI app, RAG pipeline, auth, SQLite, scoping
  web/              # Next.js frontend (V2)
  app/legacy_v1/    # retired Streamlit UI, kept for reference
  data/             # test PDFs; data/users/<id>/ for uploads (git-ignored)
  chroma_store/     # persisted vector DB (git-ignored)
  smartdoc.db       # SQLite relational store (git-ignored)
  docs/             # pipeline dossiers
  eval/ scripts/ tests/
  .env              # real secrets (git-ignored)
  .env.example      # documents variable names only
  requirements.txt
  README.md  DECISIONS.md
```

Current status: **V2 phases 1–3 complete.** The V1 retrieval stack (PDF ingestion,
persistent Chroma indexing, intent-aware hybrid retrieval, context assembly,
grounded generation, structural citations, grounding verification/repair) is
carried over unmodified. V2 adds per-user isolation, FastAPI-owned authentication,
per-session chat memory, and the Next.js frontend.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/v2-pipeline-dossier.md`](docs/v2-pipeline-dossier.md) | **Current.** All 21 stages end to end identity, scoping, ingestion, the query pipeline, session memory, the API, and the frontend with the functions that run at each and flow diagrams. |
| [`docs/v1-pipeline-dossier.md`](docs/v1-pipeline-dossier.md) | The V1 pipeline as closed out. Stages 07–16 of the V2 dossier are this code, unmodified. |
| [`DECISIONS.md`](DECISIONS.md) | Why each decision was made, including what was measured and rejected. |

## Setup

```bash
git clone <repo-url> smartdoc
cd smartdoc

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then add your real OPENAI_API_KEY
```

Configuration lives entirely in `.env` (see `.env.example` for the full list):
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBED_MODEL`, `CHAT_MODEL`. Never
commit `.env`.

## Accounts and per-user data (V2)

Every endpoint except `/health` and `/auth/*` requires a bearer token, and each
user sees only their own documents and chats. Create the development account and
adopt the existing corpus into it:

```bash
python -m scripts.seed_dev_user
```

That prints the credentials and stamps ownership onto the already-indexed chunks
(metadata only nothing is re-embedded). Sign in:

```bash
curl -s -X POST http://127.0.0.1:8000/auth/login \
     -H 'Content-Type: application/json' \
     -d '{"email":"dev@smartdoc.local","password":"devpassword123"}'
```

Use the returned `access_token` as `Authorization: Bearer <token>` on `/upload`,
`/ask`, `/documents`, and `/sessions`. Google sign-in is available at
`/auth/google/login` once `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set;
without them, only those two routes are disabled.

## Chat sessions and memory (V2 Phase 2)

`POST /ask` is stateless unless you pass a `session_id` from `POST /sessions`:

```bash
TOKEN=...  # from /auth/login
SESSION=$(curl -s -X POST localhost:8000/sessions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"title":"leave questions"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"question\":\"How many days of annual leave do Standard band employees get?\",\"session_id\":\"$SESSION\"}"

# a follow-up in the SAME session resolves the reference via the running summary
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"question\":\"And the Executive band?\",\"session_id\":\"$SESSION\"}"
```

`GET /sessions` (default `limit=10`) returns the sidebar list, most recently
*active* session first. The running summary is rewritten by a background task
after each answer is sent it never adds to `/ask`'s response time; see
`scripts/measure_memory_latency.py` and DECISIONS.md Part 6 for the empirical
check.

## Orchestration feature flags (V2 Phase 2, Part B)

Four additional retrieval/answer behaviors exist behind flags, each OFF by
default: `DOC_LOCK_ENABLED`, `PARTIAL_ANSWER_FENCING_ENABLED`,
`PLANNER_INTENT_EXPANSION_ENABLED`, `EXHAUSTIVE_TRIGGER_ENABLED` (see
`.env.example` for what each does). Measure any one against the gold set
before trusting it:

```bash
.venv/bin/python -m scripts.eval_feature --flag DOC_LOCK_ENABLED
```

## Answer voice and structure (V2 Phase 4, Part A)

Two flags, both **ON** by default, both prompt-side only:

| Flag | Effect |
|---|---|
| `ANSWER_VOICE_ENABLED` | A warm, plain, human register. Wording only it may not add a fact, a reassurance or a caveat, and may not decorate or soften the fixed refusal sentence. |
| `ANSWER_FORMAT_ENABLED` | The model shapes each answer to its content: a **table** for comparisons and repeated-attribute lists, **bullets** (numbered when order matters) for sets and steps, **prose** for explanations and single facts. |

They default ON rather than OFF because the phase's acceptance gate is that the
behaviour is *active*. Set either `false` to revert to Phase 3 answer text; with
both off the system prompt is byte-for-byte what it was.

Nothing reformats the model's output, so neither flag can move a figure or a
citation. The client renders the markdown into shadcn table/list components
(`web/src/components/chat/answer-text.tsx`, `web/src/components/ui/table.tsx`).

Set a real `JWT_SECRET` before exposing the API the default placeholder is a
published signing key, and anyone holding it can mint a token for any account.
Note that `.env` shipping the key **present but blank** resolves to that same
placeholder; `ALLOW_INSECURE_JWT_SECRET=false` makes startup refuse instead of
warn.

Verify the security and formatting guarantees:

```bash
python -m pytest tests/ -q                 # offline: prompts, refusal, pruning, citations
./run.sh --backend-only                    # then, in another shell:
python scripts/verify_phase4.py            # 60 live checks; writes eval/phase4/
python scripts/render_check.py             # renders AnswerText and asserts on the HTML
```

Relational data (users, documents, sessions, messages) lives in `smartdoc.db`;
Chroma holds only vectors and metadata. Both are git-ignored.

## Markdown ingestion and semantic splitting (V3.1)

One flag, **OFF** by default: `MARKDOWN_INGESTION_ENABLED`. With it off,
ingestion is the V2 plain-text pipeline block-for-block and hash-for-hash.

With it on, each PDF is converted to markdown (`pymupdf4llm`) before chunking and
split in two stages `MarkdownHeaderTextSplitter` on `#`/`##`/`###` first, then
the existing `RecursiveCharacterTextSplitter` only for sections still over the
locked ~800/120 budget. Every chunk gains `heading_path`
("Employee Handbook > 3. Annual Leave Entitlement") and `section_title`, and
citations can show the section a fact came from.

The plain-text path is permanent, not deprecated: it is also the fallback when
markdown conversion fails, returns near-empty text (a scanned / image-only PDF),
or recovers materially less text than the plain-text extractor. Documents that
fall back are marked `extraction_mode = "text"` on the `documents` row and in the
upload response, so a degraded document stays identifiable.

Generated markdown is cached under `data/markdown/<user_id>/` namespaced per
owner, because two users may both have uploaded `handbook.pdf`.

```bash
# structural checks only, no API calls
python scripts/verify_v3_1.py --structural
# plus the known-answer before/after table (ingests twice, into a throwaway store)
python scripts/verify_v3_1.py
```

## Tables (V3.2)

One flag, **OFF** by default: `TABLE_AWARE_INGESTION_ENABLED`. With it off, tables
behave exactly as in V2 one pipe-delimited block per detected table, inside the
normal chunk stream.

Overlap cannot fix a table. What a table fragment is missing is not the previous
120 characters, it is the **header row** which may be an entire page away. Both
multi-page tables in this corpus break that way: the WidgetX fault codes continue
from page 7 to page 8 and the page-8 fragment has no header row at all.

With the flag on:

| Stage | What happens |
|---|---|
| Extract | `find_tables()` yields rows + column headers + page, before text chunking. Table regions stay excluded from prose. |
| Stitch | Consecutive-page fragments merge into one logical table at **ingest**, so it is never a cross-page problem at query time. |
| Chunk | Split on **row** boundaries, never mid-row, with the header block repeated in every part. |
| Tag | Every part carries `table_id`, `table_part`, `table_total_parts`, `page_range`, `table_headers`. |
| Summarise | A one-line description per table (`"Table: 3. Diagnostic Fault Codes; columns: Code, Meaning, Required action; rows: E-01, ..."`) is embedded as its own small chunk with the same `table_id`. |
| Expand | A hit on any part fetches every sibling **by metadata, not by search** (~1ms), and the whole table reaches the model. |

Past `TABLE_SIBLING_MAX_PARTS` / `TABLE_SIBLING_MAX_TOKENS`, expansion degrades to
the header block, the summary, and the parts that matched rather than evicting
every other document from the context window.

```bash
python scripts/verify_v3_2.py --structural   # 40 checks, no API calls
python scripts/verify_v3_2.py                # plus the known-answer before/after
```

## Universal metadata (V3.3)

Every chunk in the index carries the **same 15 fields** text chunks, table parts,
table summaries and plain-text fallback chunks alike. A field may be empty; it may
never be absent, because a Chroma `where` clause on a key that only some chunks
carry silently returns fewer results instead of erroring.

| Group | Fields |
|---|---|
| Provenance (Layer A, exact) | `source` `page` `chunk_index` `user_id` `heading_path` `section_title` |
| Semantics (Layer B, LLM at ingest) | `content_type` `topics` `entities` `answers_questions` |
| Tables (V3.2) | `table_id` `table_part` `table_total_parts` `table_headers` |
| Ingestion | `extraction_mode` |

Validated **at write time**, after ownership is stamped and immediately before the
chunk reaches Chroma set `CHUNK_SCHEMA_STRICT=true` to make a violation raise
rather than log.

**Split by consumer, so metadata is not a token tax.** Only `source`, `page`,
`heading_path` and `content_type` reach the prompt. The other eleven including
every LLM-generated label are read by code for filtering and routing, and never
shown to the answer model. That is also what makes "a wrong auto-tag can never
corrupt an answer" structural rather than a promise.

**What the metadata buys**

| Use | Effect |
|---|---|
| Completeness | The per-document manifest (SQLite) knows a document has 7 trainings, so an answer listing 3 is caught and regenerated instead of looking finished. |
| Precision filtering | "what does the sick leave section say?" filters by `heading_path` before searching: **396 chunks 9**. |
| Richer citations | `Employee Handbook > 4. Sick Leave, p. 12` rather than a bare page number. |
| Type-aware answers | `content_type` tells the model whether it is reading steps, a rule, a table or a definition. |

```bash
python scripts/verify_v3_3.py --structural   # 25 checks, no API calls
python scripts/verify_v3_3.py                # plus each layer measured independently
```

## Exact table values (Addendum 2)

`PARALLEL_SQL_LOOKUP_ENABLED` (default **off**). Dense retrieval finds a table by
what it is *about*; it cannot find a cell by what it *contains*. Ask for one row
out of ninety and the question's embedding is nearly equidistant from all of
them.

So at ingest every table cell is **also** written to SQLite as
`(row label, column, value)` with its filename, page and table title. At query
time an indexed SQL lookup runs **alongside** the full hybrid pipeline never
instead of it.

| | Decision 1 fire SQL? | Decision 2 trust it? |
|---|---|---|
| **When** | Before any result exists | After both results are in |
| **Stance** | Permissive fire on any hint | Strict reject on any doubt |
| **Why** | ~1ms, no API call, another thread. A wrong guess is free. | A wrong value stated authoritatively is worse than none the reader cannot tell it from a right one. |
| **Rule** | A phrase fuzzy-matches a column **or** a row label **or** the question has a numeric cue | Entity **and** column both 85, **exactly one** row back, **no** multi-answer cue (all/every/list/compare/why/explain/summarize/breakdown) |

**Merge, don't choose.** When SQL is confident the model gets *both* the exact
fact, labelled authoritative and cited, and the retrieved passages so one answer
can be exact *and* explained ("78 in English, above the 40-mark passing
threshold"). Anything less than confident is dropped silently and the passages
answer alone, which is exactly the flag-off behaviour.

**Identifiers are pinned, not fuzzed.** `WRatio("employee 45", "employee 1")` is
about 95 comfortably over the trust floor so edit distance alone would answer
row 45 with row 1's salary. Resolution tries exact match first, and fuzzy
matching only ever considers labels with the *same digit runs*.

Also supported: `MAX`/`MIN`, `COUNT`, and numeric filters ("who scored highest in
Math"), which vector search cannot do at all because no passage states the
answer. Same discipline a tie has no winner, and a column living in two tables
is not aggregable.

```bash
python scripts/verify_addendum2.py --structural   # 20 checks, no API calls
python scripts/verify_addendum2.py                # plus the measured concurrency proof
```

## Run everything

From the project root:

```bash
./run.sh
```

This starts the FastAPI backend and the Next.js frontend together, waits for
`/health` before bringing up the UI, wires the UI to the API, and stops both on
Ctrl-C. Logs go to `.logs/backend.log` and `.logs/frontend.log`.

First time only, install the frontend's dependencies:

```bash
(cd web && npm install)
```

| | |
|---|---|
| App | <http://localhost:3000> |
| API docs | <http://127.0.0.1:8000/docs> |

Variations:

```bash
./run.sh --backend-only            # API only, for curl / Swagger work
./run.sh --ui-only                 # UI only, against an API already running
API_PORT=8001 UI_PORT=3001 ./run.sh   # a second instance alongside the first
```

A non-default `UI_PORT` also needs `CORS_ALLOW_ORIGINS` updated in `.env`: the
API allows a named list of browser origins, so an unlisted port has every call
blocked by the browser before it reaches the server.

The sections below cover starting each service by hand.

## Run backend

From the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload
```

The API listens on <http://127.0.0.1:8000>. Verify it:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

Interactive docs: <http://127.0.0.1:8000/docs>

> Invoke the Python tools as `.venv/bin/python -m <module>` rather than through
> the venv's `uvicorn` console script. That script hard-codes an absolute
> interpreter path in its shebang, so it breaks if the project directory is ever
> moved. `run.sh` already does this.

## Run frontend

Next.js (App Router, TypeScript) + Tailwind + shadcn/ui, in `web/`. In a second
terminal, with the backend running:

```bash
cd web
npm install                      # first time only
cp .env.example .env.local       # NEXT_PUBLIC_API_BASE_URL -> your API
npm run dev
```

The app opens on <http://localhost:3000>. Sign in with the seeded development
account (`dev@smartdoc.local` / `devpassword123`) to reach the adopted corpus, or
create a new account, which starts empty.

```
web/src/
  app/            routes: /login, /signup, /auth/callback, /dashboard, /chat
  components/     auth provider + route guard, app shell, dashboard, chat
  lib/            api client, wire types, token storage, formatting
```

The frontend is a client of the API and nothing more. It holds no retrieval
logic, no prompt, and no authorization decision: identity is the JWT, which only
FastAPI issues and verifies, and no request it sends names a user. Data isolation
is therefore a server property the UI cannot leak across accounts because it
has no notion of "whose data this is" to get wrong.

### Google sign-in

Optional. With `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` unset, the button
renders disabled with an explanation and email/password sign-in works normally.
To enable it, set those two plus:

```bash
GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/auth/google/callback
OAUTH_SUCCESS_REDIRECT=http://localhost:3000/auth/callback
```

`OAUTH_SUCCESS_REDIRECT` must point at the frontend's callback route. Left blank,
the API returns the token as JSON and the browser shows it as raw text instead of
signing in.

### Retired Streamlit client

The V1 Streamlit UI is retired and preserved at
`app/legacy_v1/streamlit_app.py`. It predates authentication and calls `/upload`
and `/ask` without a token, so it receives 401s; it is kept for reference only.

## Streamlit demo client (`streamlit_demo/`)

A second, deliberately basic UI that satisfies the "Streamlit UI with source
citation" deliverable. It is **not** a replacement for the Next.js app: it is an
independent HTTP client of the same FastAPI backend, which is the point -- if
both work, the API's auth and isolation hold from more than the one client they
were built alongside. It contains no retrieval, ranking, or generation code.

It runs separately from `./run.sh`, and both can run at once against the same
backend.

```bash
# 1. Backend must be running (./run.sh, or just the API)
uvicorn backend.main:app --port 8000

# 2. Mint a token for the demo account (once per session; see below)
python scripts/mint_demo_token.py

# 3. Start the demo UI on http://localhost:8501
streamlit run streamlit_demo/app.py
```

### Why a mint script instead of a login form

The demo signs in as one pre-configured account and shows **no login screen**.
That account signed up through Google, so it has no password and `/auth/login`
cannot be used for it.

The obvious shortcut -- an endpoint that issues a token for a given email -- is
an authentication bypass: anything reachable over the network that hands out
credentials for an arbitrary account defeats the entire auth layer. So
`scripts/mint_demo_token.py` is a **local-only script**, never imported by
`backend/` and never registered as a route. It reads the user row from SQLite
read-only and calls the same `create_access_token()` that `/auth/login` uses
internally, so the token it produces is completely ordinary.

The token is written to `.demo_token` (git-ignored, mode 0600). When it expires,
the app says so and tells you to re-run the script rather than crashing.

Configure the account in `.env`:

```
DEMO_USER_EMAIL=your-account@example.com
DEMO_API_BASE_URL=http://127.0.0.1:8000
```

### What it does

| Tab | Contents |
|---|---|
| **Chat** | Last 10 sessions in the sidebar with a New chat button; selecting one loads its history. Answers render with `st.markdown`, so tables come back as tables and step lists as numbered lists -- the backend already decided the format. |
| **Documents** | List (name, indexed chunks, date), upload a PDF via `/upload`, delete via the real endpoint, which reports how many chunks the cascade removed. |
| **Evaluation** | Loads the most recent saved report from the existing eval pipeline, or triggers a live run. Displays the overall pass rate and the per-category table with `st.dataframe`. No scoring logic is reimplemented here. |

**Citations are the deliverable.** Every generated answer renders its sources
underneath it -- document name, page (or page range), section, and the snippet --
in an expander, visually separated from the answer text. The data comes straight
from the API response's `sources` array; nothing is inferred or reconstructed.

The sidebar shows the demo user's document and session counts as plain metrics.

### Constraints this app is held to

* **No emoji anywhere**, enforced by `tests/test_streamlit_demo.py` rather than
  by promise. The same suite asserts the app never imports pipeline code, that
  the mint script stays unreachable from any route, and that a 401 becomes an
  actionable message.
* **Default Streamlit appearance.** No custom CSS, no theming; this is the
  opposite of the Next.js app's design system by intent.

```bash
python -m pytest tests/test_streamlit_demo.py -q
```

Navigation is a sidebar radio, not `st.tabs`, for two reasons: `st.tabs` always
opens on its first tab with no way to choose (so a reload could not land on
Documents), and it renders every tab's body on every rerun, which meant the
evaluation report was fetched even while reading the chat. **Documents is the
default view on load.**

The Chat view keeps a **fixed window of 5 sessions**. Creating a sixth deletes
the least recently active one, through the ordinary `DELETE /sessions/{id}`
endpoint, so the message cascade is the server's. This is a demo-client policy
only -- the Next.js client has no such limit and the backend enforces none.

> **This cap deletes real conversations.** Clicking "New chat" repeatedly to test
> it will push genuine history out. `scripts/restore_pruned_sessions.py` can
> recover deleted sessions from SQLite's free pages (dry run by default,
> `--apply` to write, and it backs the database up first), but do not rely on
> that -- reduce `MAX_SESSIONS` pressure or test against a throwaway account.

### A crash worth knowing about (fixed)

The app used to die mid-session with a macOS "Python quit unexpectedly" dialog.
The cause was **not** Streamlit: the crash report points at
`libarrow`'s mimalloc allocator, inside `mi_thread_init` reached from
`NumPyConverter::Convert` -- PyArrow 25 converting a pandas frame to an Arrow
table on one of Streamlit's worker threads, which segfaults the whole process.
It is triggered by `st.dataframe`, so the Evaluation tab reproduced it reliably.

`app.py` therefore sets `ARROW_DEFAULT_MEMORY_POOL=system` **before anything
imports pyarrow**. The system allocator has no such thread-init path, and Arrow
is only used here to draw two small tables, so nothing is lost. Verified by
rendering the Evaluation tab repeatedly under browser churn -- the exact pattern
that previously crashed every time.

If you hit it again, check the version triple; this environment runs
`streamlit 1.41.1 / pandas 2.3.3 / numpy 2.5.1 / pyarrow 25.0.0`.

### Running the venv's binaries

This project's `.venv` was created at a different path than it now lives at, so
its console shims (`.venv/bin/streamlit`, `.venv/bin/pip`) carry a stale shebang
and fail with `bad interpreter`. `.venv/bin/python` itself is fine, so invoke
tools through it:

```bash
.venv/bin/python -m streamlit run streamlit_demo/app.py
.venv/bin/python -m pip install ...        # NOT plain `pip`, which hits system Python
```

Activating the venv and running bare `streamlit` or `pip` silently falls through
to the system Python and installs there instead.
