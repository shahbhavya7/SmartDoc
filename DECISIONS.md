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
