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
| Relational store | SQLite — users, documents, sessions, messages (V2) |
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
| [`docs/v2-pipeline-dossier.md`](docs/v2-pipeline-dossier.md) | **Current.** All 21 stages end to end — identity, scoping, ingestion, the query pipeline, session memory, the API, and the frontend — with the functions that run at each and flow diagrams. |
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
(metadata only — nothing is re-embedded). Sign in:

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
after each answer is sent — it never adds to `/ask`'s response time; see
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

Set a real `JWT_SECRET` before exposing the API — the default placeholder is a
published signing key, and anyone holding it can mint a token for any account.

Verify the isolation guarantees:

```bash
python -m pytest tests/ -q
```

Relational data (users, documents, sessions, messages) lives in `smartdoc.db`;
Chroma holds only vectors and metadata. Both are git-ignored.

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
is therefore a server property — the UI cannot leak across accounts because it
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
