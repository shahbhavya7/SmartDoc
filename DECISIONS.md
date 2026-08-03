# SmartDoc — Architecture Decisions

Original locked decisions, then every change made during the production-quality
review: the problem, why it occurred, the trade-off accepted, and why the chosen
option beat the alternatives.

---

## Part 1 — Original locked decisions (unchanged)

| Decision | Rationale |
|---|---|
| FastAPI holds all logic | One place to test; the UI stays a thin client |
| ChromaDB `PersistentClient` on disk | Survives restart; no re-embedding to serve a query |
| OpenAI `text-embedding-3-small` | Same model at index and query time, else vectors are incomparable |
| `gpt-4o-mini`, temperature 0 | Cheap, and reproducible answers make consistency testable |
| Answer only from retrieved context | The fixed refusal sentence beats a confident guess |
| Citations from retrieval metadata | Structural citations cannot be hallucinated |
| Secrets in `.env`, git-ignored | `.env.example` documents names only |

---

## Part 2 — Verified bugs

These were defects, not preferences. Each was proven with a controlled test.

### B1. Editing a document left superseded text retrievable

**Problem.** Re-ingesting an edited document that produced *fewer* chunks than
before left the old version's surplus chunks in the index permanently. Proven: a
5-chunk `policy.pdf` re-ingested as 2 chunks left count at 5, with 3 stale chunks
still retrievable and citable. Verified end to end through the API afterwards —
uploading a shorter revision moved the collection 475 → **473**, not 477, with
zero stale chunks.

**Why.** `upsert` keyed on `source:chunk_index` overwrites indices 0–1 and never
touches 2–4. Earlier idempotency testing only re-uploaded *byte-identical* files,
which cannot expose this.

**Consequence.** A repealed HR policy stays answerable as though current — the
worst failure mode a policy assistant has.

**Fix.** `ingest_documents` deletes every chunk of a source before writing, plus
a `content_hash` per document so unchanged files are skipped without embedding
calls.

**Trade-off.** Replacement is not atomic: a crash between delete and upsert
leaves the document unindexed. Accepted — a *missing* document produces a
refusal, while a *stale* document produces a confident wrong answer. Rejected
alternative: version-suffixed ids with a tombstone sweep, which needs a
background reaper and adds its own failure mode.

### B2. Context and citations had diverged

All retrieved hits went into the prompt while the citation list was filtered by a
distance margin. The model read four passages; the user was shown one. Any claim
from the other three was uncited. Filtering now happens once, upstream, and
citations are built from `AssembledContext.units_used` — exactly what entered the
prompt — so the two cannot diverge by construction.

### B3. The index was L2 while the tuned threshold assumed cosine

The collection was created without `hnsw:space`, so Chroma defaulted to `l2`
(0–4 scale) while docstrings described the values as cosine distances and a
`0.30` margin was calibrated on that misdescribed scale — within 0.01 of flipping
a real citation. `CHROMA_SPACE=cosine` is now explicit, and the fragile margin is
gone entirely, replaced by rank-based selection (RRF + reranking) that needs no
per-corpus distance constant. The collection also records its embedding model and
refuses to be queried by a different one: two 1536-dimension models are
geometrically compatible and semantically unrelated, so a swap would otherwise
return confident nonsense.

### B4. No request timeout

The OpenAI client had no timeout, so one hung connection blocked its caller
forever with no error — observed as an evaluation run that sat silent for 28
minutes, and the same hang would tie up a FastAPI worker. Timeout (45s) and
bounded retries are now configured.

---

## Part 3 — Retrieval architecture

### C1. Chunking: page-independent → document-stream, heading-aware, parent/child

*Deviates from the locked "RecursiveCharacterTextSplitter, 800/120" decision.*

Chunking ran per page, so any section spanning a page break was severed and
overlap never crossed the boundary. Measured: the old pipeline produced **0**
chunks spanning a page break — structurally impossible. The new one produces 124.

Text is now assembled into a document-level stream of structural blocks
(heading / paragraph / table) carrying page numbers, and chunking runs over that.
Blocks are kept on each chunk so a parent can be re-split into children without
flattening to text — otherwise every child inherits the parent's start page and
page attribution silently breaks (a bug caught during implementation).

Children (~350 tokens) are embedded and searched; parents (~1600) are what the
model reads. Retrieval precision wants small chunks; answer quality wants whole
sections. One fixed `CHUNK_SIZE` must trade one against the other; storing both
does not.

**Trade-off.** ~2.2× more indexed chunks, a JSON sidecar for parents, more moving
parts. Accepted: this is what lets coverage rise without noise rising with it.
**Rejected:** embedding-based semantic chunking (an embedding call per candidate
boundary at ingest, boundaries no human can predict, small gain on documents that
*have* headings); bigger fixed chunks (raises recall, destroys precision).

### C2. Tables extracted as structure

`get_text("text")` interleaves table cells into unreadable prose, destroying
every fault-code and entitlement table at ingest. PyMuPDF's table finder now
renders tables as pipe-delimited rows, and prose blocks overlapping a table's
bounding box are skipped so cells are not emitted twice. Tables are never split
mid-row — half a table loses its header row and becomes uninterpretable.

**Verified:** "list all diagnostic fault codes" returns all 9 codes across a table
split over two pages **plus** 2 conditions documented only in prose.

### C3. Running headers/footers removed by frequency

A short line near a page margin appearing on ≥60% of pages is stripped.
Frequency beats pattern-matching "Page X of Y": body sentences do not recur
verbatim across most pages.

**Caveat found in testing.** A document's title block is usually the *same string*
as its running header, so stripping deleted the title — and with it the breadcrumb
that helps every chunk. Page 1 is therefore exempt, and only the first repeating
line on page 1 may become a heading, so a footer like "Internal Use Only" cannot
become a section title (also observed).

### C4. Metadata: 3 fields → 11

`{source, page, chunk_index}` could not support parent lookup, neighbour
expansion, section grouping, or staleness detection. Added `doc_title`,
`section`, `page_end`, `parent_id`, `prev_id`, `next_id`, `has_table`,
`token_count`, `content_hash`. The `"<title> > <section>"` breadcrumb is also
prepended to embedded text: a chunk deep inside "3. Annual Leave Entitlement"
that never repeats "annual leave" is otherwise near-invisible to a query using
those words.

### C5. Hybrid retrieval and rank fusion

Dense search plus BM25. Dense generalises across paraphrase; BM25 nails rare
exact tokens — "E-07", "AES-256", "Tier 3" — where embeddings are weakest because
such tokens carry little distributional signal. The tokenizer preserves
hyphenated identifiers, since splitting "E-01" into "e" and "01" defeats the
purpose.

Fusion is **Reciprocal Rank Fusion** (k=60). Cosine distances and BM25 scores are
on incomparable scales; weighted-sum fusion needs per-corpus normalisation
constants that do not transfer, while RRF needs only ranks. **Rejected:** score
normalisation, which reintroduces exactly the tuned-constant fragility of bug B3.

### C6. Intent-aware retrieval modes

Profiles used to vary *how much* was retrieved, never *how it was found*, so
document-wide synthesis and exhaustive extraction ran the same similarity search
as a fact lookup at a bigger `k`. Both are **coverage** problems, and top-k by
similarity is the wrong primitive for coverage.

| Mode | Intent | Strategy |
|---|---|---|
| `focused` | fact lookup | few chunks, strict rerank floor, gated |
| `per_entity` | comparison | round-robin slots per entity, then merge and rerank |
| `multi_hop` | multi-step | bridge-first decomposition, **routing not enforced** |
| `outline` | synthesis | subtopic plan + section breadth, document order |
| `sweep` | exhaustive | examines **every** section of the routed documents |
| `broad` | cross-document | spread across documents, ungated |

*Per-entity* exists because a comparison fails when one entity's section
outscores the other's and takes every slot: the answer covers one side and
implies symmetry.

*Sweep* exists because similarity ranking stops at the densest occurrence.
It loads every chunk of the routed documents and reranks in batches, keeping
**all** passing sections. Only affordable because routing has already narrowed
the corpus; above `SWEEP_MAX_CANDIDATES` it degrades to keyword pre-filtering and
*reports* the truncation, since a silent cut is indistinguishable from "there was
nothing else".

### C7. Hierarchical document routing

`query → score documents → select documents → chunk retrieval → rerank`.

Documents are scored from a corpus-wide sample of fused chunk hits, summing each
document's **top few** chunk scores. Summing all hits ranks by length: a 33-page
manual accumulates more weak matches than a 6-page register that answers the
question exactly.

Selection is **relative** — a document survives at ≥ `DOC_SCORE_DROP_RATIO` of the
leader. Absolute thresholds do not transfer across corpora or embedding models;
the ratio between best and second-best is scale-free. Modes that draw heavily from
each document they keep use a stricter gate (`outline` 0.60, `sweep` 0.65).

**Measured:** a synthesis question that previously drew a quarter of its context
from an unrelated PDF now scores that PDF 0.207 against a leader at 0.90 and
excludes it.

**Trade-off, stated plainly:** routing can gate out a document the question
needed. Three mitigations: the leader is always kept, so routing never returns
nothing; `multi_hop`/`cross_document` never enforce the gate; and
`CROSS_DOC_RESERVE_SLOTS` keeps a few candidate slots for chunks *outside* the
routed set. That last one exists because gating amplified misclassification —
multi-hop recall fell 0.62 → 0.38 when routing was introduced, entirely on
questions the classifier had labelled `fact_lookup`. Routing should bias
retrieval, not blind it.

### C8. Subtopic planning

The planner returned paraphrases of the question. It now returns the **dimensions
a complete answer must cover**, derived from subject matter so it generalises.
"Design a complete workflow for onboarding a vendor" plans: vendor selection,
contract negotiation, documentation requirements, system integration, training
and support, performance evaluation, ongoing communication. Each becomes a
retrieval query prefixed with the question's subject terms, so a bare subtopic
like "testing" cannot wander into an unrelated document's testing section.

### C9. Reranking credits bridging passages

The reranker judged candidates against the original question, which structurally
penalises the passage stating a **rule** in a multi-hop chain — "Restricted data
must be encrypted with AES-256" never mentions "payroll records", so it ranked #1
in dense search and was then discarded. The prompt now awards 3 to a passage
supplying a rule, classification, or definition the answer depends on, *even when
it shares no vocabulary with the question*.

### C10. Escalation on shortfall

A refusal — or an answer reporting its own missing coverage — from a cheap
single-query plan triggers **one** retry with the multi-hop profile. Some
questions need an intermediate lookup with no surface marker saying so ("what
encryption is required for payroll records?" reads like a plain lookup). Paying
for decomposition only after a cheap attempt fails keeps the fast path fast.

A question naming a capitalised entity also never takes the no-classifier fast
path. Found the hard way: "How often must Northwind Logistics complete a SOC 2
review?" was fast-pathed, the bridging fact was never retrieved, and the model
asserted "Tier 3" **unsupported** — right answer, no evidence.

### C11. Context assembly

Blocks were emitted in distance order with no dedup and no grouping, so
overlapping chunks repeated text verbatim, one document's sections appeared out of
order, and the strongest evidence could land mid-prompt where attention is
weakest.

Near-duplicates are detected by shingle **containment** (chunk overlap produces a
*subset* relationship, which symmetric measures like Jaccard score as only
moderately similar and therefore miss). Adjacent passages are merged before the
model sees them. For synthesis and procedures, relevance-interleaving is
**deliberately disabled** and document order preserved — scrambling a policy's
sections to fight position bias destroys the sequence the answer must follow — and
the document outline is prepended so the model can state what it did not cover.

**Bug caught here.** The first implementation budgeted whole document *groups*, so
a multi-hop question retrieved its bridging fact and then lost it because a large
sibling pushed the group over budget. Budgeting is now **per unit**, ranked across
all documents, before grouping.

### C12. Prompting

Failure modes differ by intent, so instructions do too: exhaustive extraction is
told to enumerate items from tables *and* prose; comparison to address every named
entity; synthesis to follow the document's structure and say what it omits;
cross-document to attribute by document name and surface conflicts rather than
silently choosing. Refusal is graded — previously binary, which produced answers
with the refusal sentence bolted on ("Secondary caregivers get four weeks. I don't
know based on the available documents.").

### C13. Citations

Snippets were cut from the head of a chunk, so the cited text usually did not
contain the supporting sentence. A lexical-centering fix worked in English only: a
Spanish query about annual leave cited the *Sick Leave* section, because zero
English content-word overlap fell back to head-of-section. The snippet is now cut
from the **child chunk that matched** — selected by embedding similarity, hence
the semantically matched span regardless of query language.

Cross-lingual questions are also translated for retrieval, because BM25
contributes nothing in another language and dense retrieval alone picked the wrong
document (observed: a Spanish leave question retrieving the contractor handbook).

### C14. Grounding is enforced, not merely reported

Order: regenerate once with the offending claims named → re-verify → prune the
offending sentences → withdraw to the refusal if nothing substantive survives.

Two guards, both added after remediation caused real damage in testing:

* **Absence statements are never repaired.** The judge sometimes flags "the
  documents do not specify X". That is not a claim about the world. Acting on it
  deleted a legitimate hedge and, in one case, regeneration dropped a correct
  "AES-256" with it.
* **A repair may not remove supported values.** Any identifier or figure the
  context *does* support must survive the rewrite; otherwise the repair is
  rejected in favour of pruning, or the original is kept with the flag visible. A
  repair that trades a flagged answer for a less complete one is not an
  improvement.

Structural and LLM signals are reported separately: derived arithmetic ("a
difference of eight days" from 20 and 28) lands in `unverified_numbers` and must
not by itself condemn an answer, or real hallucinations drown in noise.

### C15. Embedding model kept at `text-embedding-3-small`

Deviation was authorised but the evidence did not justify it: retrieval precision
and recall for fact lookups were already at ceiling, so the failures were
structural (chunking, fusion, assembly), not representational.
`text-embedding-3-large` would add ~6.5× embedding cost to fix something that was
not broken. `EMBED_MODEL` is a config value and the collection guards against a
mismatched swap, so this is one line to revisit if the gold set ever shows a
representation ceiling.

### C16. Performance

Shared OpenAI client and cached Chroma clients (a fresh client per call meant a
fresh connection pool and a re-opened sqlite file every request); one batched
embedding call covers all sub-queries; the BM25 index is cached on
`(collection, chunk count)` so it rebuilds automatically after an upload.

---

## Part 4 — Evaluation framework

**Corpus.** 21 chunks over 7 mostly-synthetic 1–2 page PDFs meant `top_k=4`
touched 19% of the corpus per query; the hard query classes could not be
reproduced, let alone measured. Replaced by a generated corpus of 10 documents /
130 pages / ~476 chunks with planted multi-hop chains, cross-document
comparisons, exhaustive lists spanning a table *and* prose, near-duplicate
distractors (a contractor handbook mirroring employee-policy wording with
different numbers), page-spanning sections, running headers, and hyphenation
artifacts. Deterministic, so labels stay valid. The original PDFs are preserved in
`data/legacy_v1/`, outside the non-recursive ingest path.

**Labels are fact strings, not chunk ids.** The harness locates each required fact
in the parsed corpus to derive gold `(source, page)` pairs at run time, so labels
survive re-chunking and re-pagination. A gold set keyed on chunk ids rots the
moment chunk size changes, and then every retrieval metric measures the wrong
thing. Unlocatable facts are reported as **gold drift** rather than silently
scoring zero.

Two measurement bugs were found and fixed in the harness itself:

* Page-pair windows were merged with single-page hits, attributing every fact to
  both pages of every containing window. That inflated the gold set ~3× and
  dragged reported recall from 0.98 to 0.43 — a pure artifact that looked exactly
  like a retrieval regression. Resolution is now precise-first.
* The LLM correctness judge was unreproducible: asked which of 4 required facts
  were missing, it returned 7. Correctness is now scored structurally against
  labelled tokens, with the judge retained only where no token list exists.
* A single dropped connection aborted a 30-question run and discarded every
  completed measurement. Per-question failures are recorded and excluded from the
  averages.

**Metrics.** Retrieval precision@k, recall, MRR, source recall; context relevance
(token-weighted, because one large irrelevant block does more damage than three
small ones) and completeness (the ceiling on answer quality); answer correctness;
faithfulness; hallucination rate; citation coverage; intent-classification
accuracy; false and correct refusals; latency.

**Ablation.** `--ablation` disables one stage at a time (rerank, hybrid,
decomposition, parent expansion, document routing, grounding repair) so each
stage's contribution is *measured* rather than asserted. A stage that does not
move the numbers on this corpus should be called out, not defended.

---

# V2 — multi-tenant SmartDoc

Everything above is carried over unchanged. The retrieval stack — dense, BM25,
RRF, reranking, the adaptive modes, grounding — is treated as a fixed input; V2
adds ownership around it and does not alter how anything is found or ranked.

## Part 5 — Phase 1: users, auth, and isolation

### D1. Two stores, joined on `document_id`

SQLite owns `users`, `documents`, `sessions`, `messages`; Chroma keeps vectors
and metadata and nothing relational. The join key is `document_id`, stamped into
every chunk's metadata at ingest.

Deleting by *filename* was the obvious alternative and is wrong once filenames
are no longer unique: two users may both own a `handbook.pdf`, and one of them
renaming or re-uploading theirs would make the link ambiguous. An immutable id
survives both.

`PRAGMA foreign_keys = ON` is issued per connection. SQLite compiles foreign-key
support in but leaves it **off**, so the declared constraints would otherwise be
documentation — integrity that looks enforced and is not.

### D2. Isolation is enforced at one choke point, not at each call site

Every Chroma read in the system already funnels through `backend/vectorstore.py`.
Scoping is applied *there*: reads merge `{"user_id": <scope>}` into their filter,
writes stamp ownership onto metadata. The active user is bound to a
`contextvars` context by the request layer.

**Why not thread a `user_id` parameter through the pipeline.** Retrieval spans
six modules and roughly 4000 lines and is explicitly final in V2. Threading an
argument through it would mean editing the code this phase is forbidden to
touch, across dozens of call sites, where a *single* missed one is a silent
cross-tenant leak that no test failure announces. A context variable makes "no
scope bound" the only way to read unfiltered, and that state is reachable only
from maintenance scripts, never from a request.

**Why `contextvars` and not a module global.** FastAPI runs sync endpoints in a
threadpool; contextvars are copied into those workers, so two concurrent
requests from different users each carry their own scope. A global would be
shared between them — an isolation bug that appears only under load, which is
the worst kind.

**Trade-off, stated plainly.** The scope is ambient rather than explicit in each
function signature, so reading `retrieval.py` alone does not reveal that its
reads are filtered. Mitigated by three things: `vectorstore.py` is the only
module that touches the collection directly (verified — no `collection.get` or
`collection.query` survives anywhere else), `ScopeError` fails closed rather
than falling back to unfiltered access, and the isolation test asserts the
property end to end rather than trusting the mechanism.

Three direct-Chroma call sites in `routing.py` and `retrieval.py` were routed
through new scoped helpers (`get_chunks_where`, `get_chunks_by_ids`). Those
edits change *which rows* are read, never how results are scored, ordered, or
fused.

### D3. Chunk ids are namespaced per user

Ids were `"<source>:<chunk_index>"` — unique per document, **not** per user. Two
users uploading `handbook.pdf` would produce identical ids, and the second
upsert would silently overwrite the first user's chunks: data loss dressed as an
update. Ids written under a scope are now prefixed `u<user_id>|`, and the
id-shaped metadata (`id`, `parent_id`, `prev_id`, `next_id`) is prefixed with
them so neighbour and parent expansion keep resolving.

Verified live: user B uploading a `vendor_register.pdf` that the dev user
already owned indexed 16 new chunks and left the dev user's 16 untouched.

### D4. BM25 is per-user, and the cache key says so

The lexical index is a *materialised copy* of the corpus, so unlike a Chroma
query it cannot be filtered after the fact — an index built across tenants would
surface another user's passage as a keyword hit with no metadata filter in the
path to stop it. `all_chunks()` is scoped, and the cache key gained the user_id.

Eviction changed with it: the old code cleared the whole cache on every rebuild,
which with several signed-in users would make one upload invalidate everyone.
Only entries built against a stale chunk count are now dropped.

### D5. Deletion order: vectors first, then the row

A crash between the two steps has to leave *some* inconsistent state; the
question is which. Vectors-first leaves a listed document with no vectors —
unanswerable, visible, and retryable by the user. Row-first would leave
orphaned vectors with no row to delete them by: still retrievable, still
citable, and no longer visible in any UI. That is bug B1 one level up, and the
same reasoning applies — a missing document produces a refusal, a stale one
produces a confident wrong answer.

### D6. Ownership is part of the lookup, not a check after it

`get_document(user_id, document_id)` filters on both columns; there is no
`get_document(document_id)` to forget to guard. Another user's id simply does
not resolve, so the endpoint returns **404 rather than 403** — a 403 would
confirm that the id exists, letting a caller enumerate the id space.

Messages have no owner column of their own; `list_messages` joins to
`sessions.user_id`, so a guessed `session_id` returns nothing.

### D7. Auth details worth naming

* **Login failures are indistinguishable.** Unknown email and wrong password
  return the identical message; distinguishing them turns the login form into an
  account-existence oracle. Asserted in the test suite, not just intended.
* **`algorithms` is pinned on decode.** Otherwise a token declaring
  `"alg": "none"` is accepted as signed.
* **Passwords over 72 bytes are rejected, not truncated.** bcrypt silently
  ignores the excess, so two different long passwords would both authenticate —
  a password that is weaker than it looks.
* **Google accounts are matched on `sub`, not email.** `sub` is the stable
  account identifier; an email can be reassigned. An existing password account
  is linked only when Google reports the address *verified* — otherwise an
  unverified claim to an address takes over the account behind it.
* **The user row is re-read on every request** rather than trusted from the
  token body, so a deleted account stops working immediately instead of at token
  expiry.
* **`/health` no longer reports corpus counts.** It is unauthenticated, and a
  global chunk total tells an anonymous caller how much every user has uploaded.
  Per-user counts moved to `GET /documents`.

### D8. The pre-V2 corpus was adopted, not deleted or re-embedded

491 existing chunks had no owner, and an ownerless chunk is invisible to
everyone under the new rules. `scripts/seed_dev_user.py` stamps `user_id` and
`document_id` onto their metadata and registers one `documents` row per source,
making the whole existing library the dev user's. Metadata only — no vectors,
text, boundaries, or embedding calls. The 247 parent records are stamped too;
skipping them would leave parent expansion silently returning nothing, which
degrades every answer to child-only context while still appearing to work.

Legacy ids are deliberately **not** rewritten to the new prefixed scheme. Their
`prev_id`/`next_id`/`parent_id` links point at each other consistently, so
renaming would break every link to no benefit. The two id schemes coexist
because every lookup resolves by metadata, not by id shape.

Verified end to end: signed in as the dev user, "How many days of annual leave
do Standard band employees get?" still answers *twenty (20) days*, cited to
`employee_handbook.pdf` p7, through the unmodified routing → hybrid → rerank →
grounding path.

### D9. `MULTI_USER_ENABLED` defaults ON

The standing V2 rule is that new behaviour hides behind a flag defaulting OFF.
This one does not, and the exception is deliberate: Phase 1's acceptance gate is
that isolation is *active*, and a flag defaulting OFF would ship a system that
passes review by not doing the thing. Setting it false is the documented revert
to known-good V1 behaviour.

### Known breakage, accepted

The Streamlit client (`app/streamlit_app.py`) calls `/upload` and `/ask`
unauthenticated and now receives 401s. Expected: the phase brief anticipates
single-user behaviour breaking, and Streamlit is replaced by the Next.js client
in a later phase. The eval harness in `scripts/` calls the pipeline in-process
with no scope bound, so it continues to run against the full corpus.

## Part 6 — Phase 2: conversational memory and orchestration upgrades

The retrieval stack — dense, BM25, RRF, reranking, the adaptive modes,
grounding — is unchanged again this phase. Everything below either sits
strictly above it (session memory) or is a bounded, flag-gated override applied
before or after it (Part B).

### Part A — per-session memory

**E1. The summary replaces the transcript as what feeds the next turn.**
`sessions.summary` is a single condensed paragraph, rewritten on every turn by
one `UTILITY_MODEL` call that reads the OLD summary and the LATEST turn and
returns a new, bounded (`MAX_SUMMARY_CHARS` = 800) summary — not an append.
Storing and re-sending the raw transcript would grow the prompt without bound
and dilute it with small talk; asking the model to re-condense every time is
what keeps the result a fixed, small cost regardless of session length, with no
separate truncation logic needed.

**E2. Memory resolves references; it does not supply facts.** The summary is
inserted into the GENERATION prompt only, in a block labelled "Conversation
memory (for resolving references only)", and the system prompt is told
explicitly to use it only for pronouns/references and never as a source of
claims — every factual statement must still come from the retrieved context.
Without this split, a wrong or stale summary line becomes a fact the model
repeats, which defeats the whole "answer only from retrieved context"
guarantee (locked decision, Part 1).

Retrieval itself is NOT given the summary text. Only `conversation_focus` (the
filename most recently cited) reaches retrieval, through the SAME parameter
Feature 1 already used — deliberately not a new lever, so a follow-up's
retrieval quality does not depend on how well summarization went, only on
which document was last discussed.

**E3. `last_document` is set synchronously; the summary is not.** After an
answer, `sessions.last_document` is written immediately from
`response.sources[0].source` — a plain field assignment, no LLM call, so it
costs nothing on the request path. Only the summary needs an LLM round trip,
and that is the one thing deferred to the background task.

**E4. NO-LAG is `BackgroundTasks`, not a bespoke thread.** FastAPI's
`BackgroundTasks` are documented to run only after the response has been sent;
using it rather than hand-rolling `asyncio.create_task` keeps the guarantee
attached to a well-tested primitive instead of new concurrency code with its
own failure modes. Verified two ways:

* Empirically, against a live `uvicorn` server (not the in-process
  `TestClient`, which awaits background tasks as part of the same call and so
  cannot distinguish "before" from "after" the response): median latency
  stateless vs. a session's later turns differed by **+312ms** against a
  ~3000ms run-to-run noise floor on this corpus — not the several-hundred-ms-to-
  seconds cost an inline summarization call would add. Script:
  `scripts/measure_memory_latency.py`.
* The stored summary was independently confirmed to exist after the request
  returned, proving the background call genuinely ran rather than merely being
  scheduled and dropped.

**E5. A background task's exceptions are not swallowed by the framework —
proven, not assumed.** A test with an unsafe stand-in task showed Starlette
propagating a background task's exception straight through the request
(`test_background_task_exception_is_not_swallowed_by_starlette`). The actual
scheduled function, `backend.memory.summarize_turn_and_store`, therefore wraps
its own body in `try/except Exception: logger.exception(...)` — the safety net
has to be the callee's, because nothing upstream provides one. A failed
summarization degrades to "this turn wasn't remembered", never a 500.

**E6. Sidebar ordering is "last active", not "last created".** `GET /sessions`
(default `limit=10`) orders by `COALESCE(MAX(messages.created_at),
sessions.created_at)` — a session replied to five minutes ago outranks one
created an hour ago with no reply. A dedicated `last_activity` column, updated
on every message insert, was rejected: at this schema's scale a `LEFT JOIN` +
`GROUP BY` over the existing `idx_messages_session` index is already fast and
needs no write-path bookkeeping to keep in sync.

**Bug found and fixed while building this.** `messages` were ordered by
`(created_at, id)`. `created_at` has second resolution, and a question and its
own answer are routinely stored within the same second — at which point `id`
(a random UUID) decided the order, silently reversing a real turn roughly half
the time. Reordered to `(created_at, rowid)`: SQLite's implicit, monotonically
increasing insertion-order column, which needed no schema change. Caught by a
Phase 2 test asserting message order, not by anything in Phase 1, which only
asserted message *count*.

### Part B — orchestration upgrades (each flag OFF by default)

**E7. Feature 5, document lock, is a new binary decision, not a bigger
Feature-1 weight.** Feature 1 (`ROUTER_ENABLED`) already nudges a document's
*score*, bounded by `ROUTER_SIGNAL_WEIGHT` and floor-protected by
`ROUTER_PROTECT_RATIO` specifically so it can never overturn strong retrieval
evidence. A genuine lock needs the opposite property on the cases it fires for:
when the user names one document, or refers to "this document" with a
conversation focus already established, retrieval should restrict to exactly
that document regardless of the ratio-based gate — a user's explicit reference
outranks a similarity score. `backend.doc_router.detect_lock` is therefore a
separate, conservative decision function, not a tuning of Feature 1's weight:

* Fires only when the question reads as ambiguous-free — exactly one scored
  document's title/filename terms are a subset of the question's terms, or a
  bare reference resolves to an existing `conversation_focus`. Naming two
  documents, or naming none with no focus set, does not lock.
* Never fires on a question that reads as a comparison (`_COMPARE_INTENT_RE`):
  locking during "compare X and Y" would silently drop the second side, which
  is the exact failure the per-entity retrieval mode exists to prevent.
* Never fires for `multi_hop`/`cross_document` intents, whose evidence lives in
  a document the user did not name by definition (Part 3, C7's
  `CROSS_DOC_RESERVE_SLOTS` exists for exactly this reason) — locking there
  would delete the bridging passage.
* When it fires, `CROSS_DOC_RESERVE_SLOTS` still runs against the locked
  document exactly as it does against a routing-gated one, so the same safety
  valve against misclassification applies.

**E8. Feature 6, partial-answer fencing, targets a real visibility gap, not a
missing behaviour.** The base prompt already tells the model to answer the
answerable part and name what's missing (Part 1's graded-refusal design, C12).
The gap was narrower and one level lower: when `enforce_grounding` finds an
unsupported claim that shares a sentence with a supported one, pruning would
cost the supported content, so remediation "declines" and returns the ORIGINAL
answer — with the unverified claim visible only in
`grounding.unsupported_claims`, metadata a UI is not guaranteed to render. With
the flag on, that specific path appends a clearly delimited
`[Unverified — not confirmed by the documents: ...]` block to the answer TEXT
itself. The prose is untouched — still a normal-looking answer — but a caller
that only renders `answer` can no longer mistake the flagged claim for a
grounded one. Scoped to exactly the "declined" branch: it changes nothing for
an answer that verifies cleanly or repairs cleanly.

**E9. Feature 7 (planner intent expansion) and Feature 8 (exhaustive trigger)
are post-classification overrides, not new heuristic-path signals.** The
existing `heuristic_type()` regexes (`_SYNTHESIS_RE`, `_EXHAUSTIVE_RE`) only
gate the FAST heuristic path and the fallback — the common case for any
non-trivial question is the LLM classifier, which never consults them. Adding
these signals there would have no effect on most real questions. Both
features are instead applied by `_apply_lexical_overrides`, called at every
`analyze()` return site (fast path, LLM path, and fallback alike), so they can
override what the classifier decided:

* Feature 7 upgrades `fact_lookup`/`procedural` to the `synthesis` profile
  (outline mode) when the question names a whole-document artifact type
  (guide, playbook, checklist, manual, SOP, onboarding, policy, journey,
  timeline) — "what's in the onboarding guide?" names an artifact rather than a
  fact, and a plain-lookup profile under-retrieves it. Never downgrades a type
  already scored as something more specific (comparison, multi-hop,
  exhaustive, cross-document, synthesis itself).
* Feature 8 forces the `exhaustive` profile (a full sweep) on "all X", "every
  X", or "everything" regardless of classification, and is checked FIRST — a
  question can trip both regexes ("list every SOP requirement"), and
  completeness is the stronger of the two claims, so upgrading to synthesis
  afterward would be a downgrade.
* Neither feature re-runs LLM decomposition for the upgraded type: `sub_queries`
  keeps whatever the original classification produced (typically just the bare
  question), and `retrieve()`'s `plan.sub_queries or [plan.question]` fallback
  already handles that — the retrieval MODE changes (outline/sweep instead of
  focused), which is what each feature is actually for, even without the
  extra decomposed sub-queries a full synthesis/exhaustive classification
  would have produced.

**Measurement.** `scripts/eval_feature.py` already existed for exactly this
purpose (flag OFF vs. flag ON over the labelled gold set, regression as a
non-zero exit code) — no new harness was built. Results for all four Part B
flags, full 30-question gold set, both passes:

| Feature | Answer correctness | Retrieval precision | Faithfulness | Median latency | Verdict |
|---|---|---|---|---|---|
| `DOC_LOCK_ENABLED` | 0.852 → 0.852 (=) | 0.622 → 0.618 (−0.004) | 0.962 → 0.926 (−0.036) | 7719 → 7180ms (−540) | **No regressions** |
| `PARTIAL_ANSWER_FENCING_ENABLED` | 0.889 → 0.889 (=) | 0.663 → 0.640 (−0.023) | 1.000 → 1.000 (=) | 8992 → 8193ms (−799) | **No regressions** (2nd run) |
| `PLANNER_INTENT_EXPANSION_ENABLED` | 0.889 → 0.852 (−0.037) | 0.621 → 0.579 (−0.042) | 1.000 → 1.000 (=) | 7866 → 8423ms (+558) | 1 flagged, did not reproduce |
| `EXHAUSTIVE_TRIGGER_ENABLED` | 0.889 → 0.815 (−0.074)* | 0.613 → 0.577 (−0.036)* | 0.963 → 0.963 (=) | 8919 → 7990ms (−929) | 1 flagged, does not reproduce |

\* First measurement, before a fix described below; see that row's note.

**Every flag was run through `scripts/eval_feature.py` at least twice** (the
full 30-question gold set, both passes), because the first pass surfaced a
real methodological problem: this corpus contains at least two questions —
`hop-northwind-review` and `exh-tier3-vendors` — whose correctness varies
run-to-run at temperature 0, independent of any flag. Proven by running each
flagged question several times, flag OFF and flag ON, outside the paired
harness: `hop-northwind-review` was wrong on an OFF run and right on the
matching ON run in one probe, then wrong on ON and right on OFF in a *later,
unrelated* flag's run. A single paired OFF/ON comparison cannot tell that
apart from a real regression; repeating the comparison can. **This is a gap in
the harness at n=30 with single-sample passes, not a property of any Part B
feature** — worth fixing in the harness itself (repeated sampling on
borderline questions) before trusting a single `eval_feature` run on a
near-tied result.

**`DOC_LOCK_ENABLED` (Feature 5).** Clean on the first run: zero questions
flipped from correct to wrong. Faithfulness dipped slightly (0.962→0.926) with
no matching correctness loss — the questions affected were already answered
correctly either way; locking changed which passages were read without
changing the answer. Latency improved (candidates narrow to one document
instead of running the ratio-gate's document scoring against every
candidate). **Shippable behind its flag.**

**`PARTIAL_ANSWER_FENCING_ENABLED` (Feature 6).** The first run flagged two
questions (`hop-northwind-review`, `hop-medical-records-mfa`) flipping from a
correct answer to the literal refusal string. This did not reproduce on a
second full run (clean), and does not reproduce in isolation: repeatedly
calling `evaluate_question` for both, flag OFF and ON, in the same process
found `hop-medical-records-mfa` correct in both states every time, and
`hop-northwind-review` failing via `grounding.repaired == "regenerated"` (an
entirely different code path than this feature touches) on an OFF run. This
matches the code: `_fence_unsupported` only appends text inside the single
"declined" branch of `enforce_grounding`, is only reachable when
`grounding.faithful is False`, and always returns non-refusal text — there is
no path from this feature to `_is_refusal(answer_text)` becoming true.
**Shippable behind its flag.**

**`PLANNER_INTENT_EXPANSION_ENABLED` (Feature 7).** Two separate runs each
flagged exactly one question, and it was a DIFFERENT question each time
(`exh-tier3-vendors`, then `hop-northwind-review`) — the second of which the
feature's own guard (`plan.query_type in (FACT_LOOKUP, PROCEDURAL)`) makes
structurally impossible to affect, since that question classifies as
`multi_hop`. Both are consistent with corpus noise, not a causal defect. No
structural mechanism by which this feature could cause either failure was
found. **Provisionally shippable behind its flag** — re-measure before
enabling by default in any environment, given the corpus's noise floor.

**`EXHAUSTIVE_TRIGGER_ENABLED` (Feature 8) — a real bug, found and fixed.**
Unlike the other three, the first run's regressions had a clear, reproducible,
structural cause: the override had no type guard, so it forced the exhaustive
profile's GATED sweep (`restrict_documents=True, max_documents=2`) onto
`syn-leave-overview` (needs `synthesis`'s outline breadth) and
`xdoc-leave-across-policies` (needs `cross_document`'s deliberately UNGATED
retrieval — the same reason Feature 5's lock excludes it, see E7 above). Both
regressed identically on a second, independent run — reproducible, not noise.
**Fixed** by adding the same guard Feature 7 already had:
`plan.query_type in (FACT_LOOKUP, PROCEDURAL)`, so the override can no longer
touch a question already classified as something needing a different
strategy. Re-measured after the fix: both structural regressions are gone; the
sole remaining flagged question (`exh-tier3-vendors`) is one of the two
identified noisy questions above, and shows wrong under BOTH flag states in
isolation testing (not a flip caused by this feature). **Shippable behind its
flag, with the guard.**

No flag's default changed as a result of any of this: Part B's rule is OFF by
default regardless of measurement, since the point of the flag is to keep the
system revertible to known-good, not to promote a feature once it measures
clean. Raw output for every run: `eval/phase2/*.log` and `*.json`
(git-ignored, like the rest of `eval/`).
