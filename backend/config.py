"""Central configuration, loaded from environment variables / .env.

Every tunable (chunking, retrieval, models, store location, budgets) lives
here so no module hardcodes them. Values are read at import time; modules that
need runtime overridability read them via ``import backend.config as config``
and reference ``config.NAME`` rather than binding the value at import (see
``backend.ingestion`` for why that matters).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str, default: str) -> list[str]:
    """Comma-separated list, blanks dropped. Used for the CORS allowlist."""
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Cheap auxiliary calls (query classification, reranking, grounding checks) run
# on their own model setting so the answer model can be upgraded independently
# of the plumbing.
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gpt-4o-mini")

# Low temperature everywhere: answers must be reproducible for the consistency
# check, and the auxiliary calls are classification tasks.
CHAT_TEMPERATURE = _float("CHAT_TEMPERATURE", 0.0)

# Per-request timeout and retry budget for the OpenAI client. Without an
# explicit timeout the SDK waits indefinitely, so a single hung connection
# blocks the calling thread forever -- which stalled an evaluation run for 28
# minutes with no error, and would hang a FastAPI worker in production.
REQUEST_TIMEOUT_SECONDS = _float("REQUEST_TIMEOUT_SECONDS", 45.0)
REQUEST_MAX_RETRIES = _int("REQUEST_MAX_RETRIES", 3)

# `colpali` branch experiment ONLY. Which pipeline answers /ask by default:
# "hybrid" (the text pipeline this whole config module otherwise describes)
# or "colpali" (colpali_experiment/answer.py, an entirely separate module).
# Plain os.getenv with no validation, the same precedent as EMBED_MODEL/
# CHAT_MODEL above -- there is no existing restricted-choice config helper in
# this module, and one is not worth adding for a single two-value switch.
# /ask can override this per-request (query param), so switching backends for
# side-by-side testing never requires a restart; only changing the .env
# default does, which the brief accepts.
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "hybrid")

# ---------------------------------------------------------------------------
# Chunking
#
# Small children are what get embedded and searched; large parents are what the
# model actually reads. Retrieval precision wants small chunks (a ~350-token
# chunk about one topic embeds cleanly); answer quality wants large context (a
# 350-token chunk usually cuts a section in half). Storing both -- searching the
# child, reading the parent -- gets both, which a single fixed CHUNK_SIZE
# cannot.
# ---------------------------------------------------------------------------
CHILD_CHUNK_SIZE = _int("CHILD_CHUNK_SIZE", 350)
CHILD_CHUNK_OVERLAP = _int("CHILD_CHUNK_OVERLAP", 60)
PARENT_CHUNK_SIZE = _int("PARENT_CHUNK_SIZE", 1600)

# Retained for the original single-granularity path and the chunk-size sweep.
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

# A line repeated on at least this fraction of a document's pages is treated as
# a running header/footer and removed. 0.6 is deliberately conservative: a
# genuine sentence almost never recurs verbatim on 60% of pages, while headers
# recur on ~100%.
HEADER_FOOTER_MIN_PAGE_RATIO = _float("HEADER_FOOTER_MIN_PAGE_RATIO", 0.6)

# ---------------------------------------------------------------------------
# V3.1 -- markdown ingestion and semantic (heading-aware) splitting
#
# OFF by default. With this flag off, ingestion runs the V2 plain-text path
# byte-for-byte: the same PyMuPDF block extractor, the same cleaner, the same
# chunker, and no new metadata keys with non-empty values.
#
# When ON, a PDF is first converted to markdown (``pymupdf4llm``), then split in
# two stages -- headings first, character splitter only for sections still over
# budget -- so a chunk boundary lands on a real section break instead of an
# arbitrary character count. The plain-text path stays permanently available and
# is also the automatic fallback for a scanned / image-only PDF.
# ---------------------------------------------------------------------------
MARKDOWN_INGESTION_ENABLED = _bool("MARKDOWN_INGESTION_ENABLED", False)

# Heading levels the first splitting stage cuts on. #### and deeper stay inside
# their parent section: past three levels a "section" is usually a single
# sentence, and a one-sentence chunk retrieves worse than the paragraph holding
# it.
MARKDOWN_HEADER_LEVELS = _int("MARKDOWN_HEADER_LEVELS", 3)

# A section longer than this is handed to the existing RecursiveCharacterText-
# Splitter; a section at or under it is kept whole regardless of how short. This
# is the LOCKED ~800/120 chunk budget (CHUNK_SIZE / CHUNK_OVERLAP below), not a
# new tunable -- named separately only so the two-stage splitter's threshold is
# readable at the call site.
MARKDOWN_SECTION_MAX_TOKENS = _int("MARKDOWN_SECTION_MAX_TOKENS", 0)  # 0 -> CHUNK_SIZE

# Below this many characters of markdown, conversion is treated as failed and the
# plain-text path takes over. A scanned PDF yields a handful of stray glyphs, not
# nothing, so "empty" has to mean "near-empty" to catch it. Also compared against
# the plain-text extraction: markdown is rejected if it recovers less than
# MARKDOWN_MIN_TEXT_RATIO of what the text path found.
MARKDOWN_MIN_CHARS = _int("MARKDOWN_MIN_CHARS", 200)
MARKDOWN_MIN_TEXT_RATIO = _float("MARKDOWN_MIN_TEXT_RATIO", 0.5)

# Where generated markdown is cached, so re-chunking experiments and table
# inspection do not re-parse the PDF. Namespaced per owner: two users may both
# have uploaded ``handbook.pdf`` and one must never read the other's text.
MARKDOWN_DIR = PROJECT_ROOT / os.getenv("MARKDOWN_DIR", "data/markdown")

# ---------------------------------------------------------------------------
# V3.2 -- table-aware ingestion and sibling expansion
#
# OFF by default. With this flag off, tables are handled exactly as V2 does:
# PyMuPDF's table finder renders each table as a pipe-delimited block that is
# kept whole inside the normal chunk stream, and a table spanning a page break is
# two unrelated chunks, the second of which has no header row.
#
# When ON, tables leave the text stream entirely: they are extracted as
# structured objects, stitched across page breaks into one logical table, chunked
# on ROW boundaries with the column headers repeated in every part, and given a
# shared ``table_id``. Retrieval then reassembles the whole table from a hit on
# any single part -- by metadata fetch, not a second similarity search.
# ---------------------------------------------------------------------------
TABLE_AWARE_INGESTION_ENABLED = _bool("TABLE_AWARE_INGESTION_ENABLED", False)

# Token budget per table part. 0 means "use CHUNK_SIZE" -- the LOCKED ~800/120
# budget. There is no overlap setting: overlap is meaningless for rows, and
# repeating the header block in every part is what a row fragment actually needs.
TABLE_CHUNK_MAX_TOKENS = _int("TABLE_CHUNK_MAX_TOKENS", 0)  # 0 -> CHUNK_SIZE

# Ceilings on sibling expansion. A hit on one part of a 40-part table must not
# push 40 parts into the prompt and evict every other document. Past either cap,
# the prompt gets the header block, the generated one-line summary, and the parts
# that actually matched -- see backend/tables.py.
TABLE_SIBLING_MAX_PARTS = _int("TABLE_SIBLING_MAX_PARTS", 6)
TABLE_SIBLING_MAX_TOKENS = _int("TABLE_SIBLING_MAX_TOKENS", 3000)

# Row labels (first-column values) listed in a table's summary chunk. Labels are
# what a natural-language question names ("what does E-04 mean", "the anti-bribery
# deadline"), so they are what makes a table findable. Cell VALUES are never put
# in a summary: a summary is generated text, and an answer must never be able to
# take a figure from it.
TABLE_SUMMARY_MAX_LABELS = _int("TABLE_SUMMARY_MAX_LABELS", 12)

# ---------------------------------------------------------------------------
# V4 -- grouped-JSON table chunk storage
#
# Shipped ON: scripts/migrate_grouped_json_tables.py has run against the real
# corpus and scripts/verify_grouped_json_tables.py's known-answer before/after
# showed zero regressions (see eval/v4_grouped_json/). The flag remains so the
# mechanism can still be measured against the legacy one it replaced -- the
# legacy pipe-format builder (backend.tables.build_table_documents) stays
# available for exactly that comparison, exercised by tests/test_v3_2_tables.py
# and scripts/verify_v3_2.py, both of which pin this flag OFF regardless of the
# default below. This replaces
# the pipe-delimited row-chunk text AND the separate summary-only chunk with one
# document per GROUP of rows: each row becomes a JSON object keyed by column
# header, several rows are grouped into one JSON array, and a short
# natural-language description of the table is prepended to every group. Column
# names travel inside every row object, so no header block needs repeating, and
# the description makes every group self-describing for embedding on its own.
#
# The SQL exact-lookup path (table_store.py / PARALLEL_SQL_LOOKUP_ENABLED) is
# untouched by this flag -- it reads the same LogicalTable objects independently
# and keeps storing rows relationally in SQLite regardless of how the vector
# side chunks them.
# ---------------------------------------------------------------------------
GROUPED_JSON_TABLE_CHUNKS_ENABLED = _bool("GROUPED_JSON_TABLE_CHUNKS_ENABLED", True)

# Target rows per JSON group. Not a hard cap: a group still flushes early if
# adding the next row would exceed the token budget below, so wide rows still
# keep each chunk near the ~800-token budget rather than overshooting it.
TABLE_ROWS_PER_CHUNK = _int("TABLE_ROWS_PER_CHUNK", 10)

# ---------------------------------------------------------------------------
# V3.3 -- the universal metadata layer
#
# Metadata is a headline feature of V3, not an optional add-on, so unlike every
# other V3 flag these ship **ON**. The flags remain so each layer can still be
# measured against itself (flag-OFF vs flag-ON on the known-answer set), which is
# how "it actually helped" stays a measurement rather than an assertion -- but the
# default is the shipped state, not the reverted one.
#
# Every chunk carries the complete schema in backend/chunk_schema.py regardless of
# these flags. What a flag changes is how much of it is POPULATED:
#   * Layer A off is not possible -- it is free and structural.
#   * Layer B off leaves content_type="other" and the three text fields empty.
#   * Layer C off leaves the manifest unread (it is still built).
# ---------------------------------------------------------------------------

# Layer B: one batched gpt-4o-mini call per group of chunks at INGEST, filling
# content_type / topics / entities / answers_questions. Query time is untouched.
SEMANTIC_METADATA_ENABLED = _bool("SEMANTIC_METADATA_ENABLED", True)

# Chunks per extraction call. Larger batches are cheaper per chunk but degrade:
# the model starts describing the batch instead of each passage.
SEMANTIC_METADATA_BATCH_SIZE = _int("SEMANTIC_METADATA_BATCH_SIZE", 8)

# Characters of each chunk sent for labelling. A chunk's opening states its topic;
# paying to send the tail buys little.
SEMANTIC_METADATA_CHARS = _int("SEMANTIC_METADATA_CHARS", 1200)

# Layer C: read the manifest to drive enumeration questions and heading filters.
MANIFEST_ROUTING_ENABLED = _bool("MANIFEST_ROUTING_ENABLED", True)

# An enumeration needs at least this many manifest items before the manifest
# overrides ordinary retrieval. One item is not a set.
MANIFEST_MIN_ITEMS = _int("MANIFEST_MIN_ITEMS", 2)

# Ceiling on items one enumeration may drive retrieval for. Past this the answer
# reports how many of how many it is covering rather than silently truncating.
MANIFEST_MAX_ITEMS = _int("MANIFEST_MAX_ITEMS", 25)

# When an answer covers fewer manifest items than exist, regenerate ONCE naming
# what it missed. This is the anti-silent-incompleteness step: an exhaustive answer
# returning 3 of 7 reads exactly like one returning 7.
MANIFEST_COVERAGE_REPAIR = _bool("MANIFEST_COVERAGE_REPAIR", True)

# Use 2: narrow the candidate pool by heading_path BEFORE searching when a question
# names a section ("what does the eligibility section say?").
HEADING_FILTER_ENABLED = _bool("HEADING_FILTER_ENABLED", True)

# A heading filter that matches almost everything has not narrowed anything and
# only risks excluding a relevant chunk, so it is dropped above this fraction of
# the user's corpus.
HEADING_FILTER_MAX_SHARE = _float("HEADING_FILTER_MAX_SHARE", 0.5)

# Raise on a chunk that violates the metadata contract instead of logging. Off in
# production -- refusing to index a chunk over a missing optional label trades a
# complete index for a tidy one -- and ON in tests and the verification script.
CHUNK_SCHEMA_STRICT = _bool("CHUNK_SCHEMA_STRICT", False)

# ---------------------------------------------------------------------------
# Addendum 2 -- parallel SQL + vector table retrieval
#
# OFF by default. When ON, every table's cells are ALSO stored relationally in
# SQLite at ingest, and a query that looks like it names a table cell fires an
# indexed SQL lookup **alongside** the full hybrid pipeline rather than instead of
# it. Vector retrieval always runs; SQL is speculative and is used only when it is
# confidently correct.
#
# The asymmetry between the two decisions is the whole design:
#   * Decision 1 (fire?) is PERMISSIVE -- a local indexed read costs ~1ms and no
#     API call, and a wrong guess is free because retrieval runs regardless.
#   * Decision 2 (trust?) is STRICT -- a wrong exact value stated authoritatively
#     is worse than no exact value at all.
# ---------------------------------------------------------------------------
PARALLEL_SQL_LOOKUP_ENABLED = _bool("PARALLEL_SQL_LOOKUP_ENABLED", False)

# Decision 1's fuzzy floor: how closely a query n-gram must resemble a known
# column name or row entity before SQL is worth firing. Deliberately below the
# trust threshold -- this gate decides whether to SPEND 1ms, not whether to
# believe an answer.
PARALLEL_SQL_FIRE_THRESHOLD = _int("PARALLEL_SQL_FIRE_THRESHOLD", 78)

# Decision 2's floor, applied to BOTH the entity and the column match. The brief
# fixes this at 85; it is exposed so the strictness can be measured, not tuned by
# feel.
PARALLEL_SQL_TRUST_THRESHOLD = _int("PARALLEL_SQL_TRUST_THRESHOLD", 85)

# Extension: MAX/MIN/COUNT and simple numeric filters, which vector search is
# genuinely poor at ("who scored highest in Math"). Same Decision 2 discipline.
PARALLEL_SQL_AGGREGATES_ENABLED = _bool("PARALLEL_SQL_AGGREGATES_ENABLED", True)

# A filter aggregate ("who scored above 80") legitimately returns a SET, so the
# single-value rule cannot apply to it. This cap replaces it: past this many rows
# the result is a report, not a fact, and is discarded in favour of the passages.
PARALLEL_SQL_MAX_FILTER_ROWS = _int("PARALLEL_SQL_MAX_FILTER_ROWS", 20)

# Cells stored per document. A pathological 10k-cell table would otherwise make
# every ingest and every cache warm slower for a lookup nobody asks.
PARALLEL_SQL_MAX_CELLS_PER_DOC = _int("PARALLEL_SQL_MAX_CELLS_PER_DOC", 20000)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_DIR", "chroma_store")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "smartdoc")

# Explicit distance metric. Chroma's default is "l2"; with unit-norm OpenAI
# embeddings l2 ranks identically to cosine, but the distance SCALE differs
# (0-4 vs 0-2), which silently invalidates any tuned distance threshold. Naming
# it here makes the scale a stated contract rather than an inherited default.
CHROMA_SPACE = os.getenv("CHROMA_SPACE", "cosine")

# ---------------------------------------------------------------------------
# Retrieval
#
# TOP_K is no longer a single global constant applied to every question -- it is
# the DEFAULT and the fallback. Per-query-type budgets live in
# backend/query_analysis.py, because the right k for "how many sick days?" and
# for "list every fault code" differ by an order of magnitude.
# ---------------------------------------------------------------------------
TOP_K = _int("TOP_K", 6)

# Candidates each retriever pulls before fusion/reranking. Recall is cheap here
# and precision is restored downstream by the reranker, so this sits well above
# the final k.
CANDIDATE_K = _int("CANDIDATE_K", 40)

# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
# and behaves well without per-corpus tuning: it damps the influence of deep
# ranks while keeping the top few ranks decisive.
RRF_K = _int("RRF_K", 60)

# Diversity cap applied after fusion. Without it, a comparison question can fill
# every slot from the single best-matching section and never retrieve the
# counterpart.
MAX_CHUNKS_PER_SOURCE = _int("MAX_CHUNKS_PER_SOURCE", 4)

# Token ceiling for assembled context. Bounds cost and latency, and keeps the
# prompt inside the range where models attend reliably.
MAX_CONTEXT_TOKENS = _int("MAX_CONTEXT_TOKENS", 6000)

# ---------------------------------------------------------------------------
# Hierarchical retrieval (document routing)
#
# Chunk search runs against the whole corpus unless a document-level pass
# narrows it first. Without that pass an unrelated document wins slots on loose
# vocabulary overlap, because no stage ever asks which document the question is
# about -- measured as 0.24 retrieval precision on synthesis questions.
# ---------------------------------------------------------------------------
ENABLE_DOC_ROUTING = _bool("ENABLE_DOC_ROUTING", True)

# Chunks sampled corpus-wide to score documents. Independent of the per-query
# candidate budget: routing wants breadth across documents, not depth in one.
DOC_ROUTING_CANDIDATES = _int("DOC_ROUTING_CANDIDATES", 60)

# A document is kept only if its score is at least this fraction of the top
# document's. Relative, not absolute: absolute thresholds do not transfer across
# corpora or embedding models, whereas the ratio between best and second-best is
# scale-free.
DOC_SCORE_DROP_RATIO = _float("DOC_SCORE_DROP_RATIO", 0.45)

# Chunks per document that contribute to its routing score. Bounded so a long
# document cannot outrank a precise short one on sheer volume of weak hits.
DOC_SCORE_TOP_CHUNKS = _int("DOC_SCORE_TOP_CHUNKS", 3)

# Candidate slots reserved for chunks OUTSIDE the routed documents on gated
# modes. Routing should bias retrieval, not blind it: when a question is
# misclassified as a simple lookup, hard gating deletes the bridging document
# and the answer is either wrong or a false refusal. Measured: multi-hop recall
# fell 0.62 -> 0.38 when gating was introduced, entirely on questions the
# classifier had labelled fact_lookup. A few reserved slots let the reranker
# still see the bridge.
CROSS_DOC_RESERVE_SLOTS = _int("CROSS_DOC_RESERVE_SLOTS", 3)

# Ceiling on sections examined by an exhaustive sweep before falling back to
# keyword pre-filtering. Guards against a 500-page manual.
SWEEP_MAX_CANDIDATES = _int("SWEEP_MAX_CANDIDATES", 60)

# Candidates per reranking call. Batching keeps a large sweep within a single
# request's token budget while still scoring every candidate.
RERANK_BATCH_SIZE = _int("RERANK_BATCH_SIZE", 30)

# ---------------------------------------------------------------------------
# Pipeline feature flags. Each stage can be disabled independently so the
# evaluation harness can measure its contribution in isolation (ablation)
# rather than asserting that it helps.
# ---------------------------------------------------------------------------
ENABLE_HYBRID = _bool("ENABLE_HYBRID", True)
ENABLE_RERANK = _bool("ENABLE_RERANK", True)
ENABLE_DECOMPOSITION = _bool("ENABLE_DECOMPOSITION", True)
ENABLE_PARENT_EXPANSION = _bool("ENABLE_PARENT_EXPANSION", True)
ENABLE_GROUNDING_CHECK = _bool("ENABLE_GROUNDING_CHECK", True)

# ---------------------------------------------------------------------------
# Grounding remediation
#
# Detecting an unsupported claim and returning it anyway defeats the purpose of
# detecting it. When verification fails the answer is regenerated with the
# offending claims named for removal; if that still fails, the unsupported
# sentences are excised.
# ---------------------------------------------------------------------------
ENABLE_GROUNDING_REPAIR = _bool("ENABLE_GROUNDING_REPAIR", True)
MAX_GROUNDING_REPAIRS = _int("MAX_GROUNDING_REPAIRS", 1)

# Requested alias for the flag above, so the orchestration layer can be
# configured with one naming convention. Repair itself is unchanged.
GROUNDING_REPAIR_ENABLED = _bool("GROUNDING_REPAIR_ENABLED", ENABLE_GROUNDING_REPAIR)

# ---------------------------------------------------------------------------
# Orchestration layer (additive, all OFF by default)
#
# These gate NEW behaviour layered on top of retrieval. Retrieval itself --
# dense embeddings, BM25, RRF, hybrid search, chunk sizes, the reranker, the
# adaptive modes, decomposition, and the grounding verifier's DETECTION logic --
# is unchanged and is treated here as a fixed input.
#
# With every flag below OFF, the pipeline behaves exactly as it does today.
# ---------------------------------------------------------------------------

# Feature 1: extra document-routing signals (title similarity, explicit
# references, conversation focus, entity overlap) layered over the existing
# similarity-derived document scores.
ROUTER_ENABLED = _bool("ROUTER_ENABLED", False)

# How much the additional signals may move a document's routing score, as a
# multiplier on the existing score. Bounded so the retriever's own evidence
# stays dominant: the brief is to bias toward precision WITHOUT dropping true
# positives, and an unbounded bonus could reorder documents the retriever
# strongly supports.
ROUTER_SIGNAL_WEIGHT = _float("ROUTER_SIGNAL_WEIGHT", 0.35)

# A document the retriever supports this strongly is never demoted by the extra
# signals -- the protection against dropping true positives.
ROUTER_PROTECT_RATIO = _float("ROUTER_PROTECT_RATIO", 0.85)

# Feature 2: recommendation planner for combination/workflow questions.
PLANNER_ENABLED = _bool("PLANNER_ENABLED", False)

# Feature 3: outline-driven synthesis guaranteeing section coverage.
OUTLINE_SYNTHESIS_ENABLED = _bool("OUTLINE_SYNTHESIS_ENABLED", False)

# ---------------------------------------------------------------------------
# Phase 2, Part B -- further orchestration features. Same rule as above: OFF by
# default, measured flag-ON vs flag-OFF on the gold set before being trusted.
# ---------------------------------------------------------------------------

# Feature 5: hard-lock retrieval to a single named/focused document instead of
# merely biasing toward it (Feature 1/ROUTER_ENABLED). Never applied to
# multi_hop or cross_document questions -- see backend/doc_router.detect_lock.
DOC_LOCK_ENABLED = _bool("DOC_LOCK_ENABLED", False)

# Feature 6: when a grounding failure can't be safely repaired without losing
# supported content, visibly fence the unverified part in the answer text
# itself instead of returning the flagged answer unmarked to the reader.
PARTIAL_ANSWER_FENCING_ENABLED = _bool("PARTIAL_ANSWER_FENCING_ENABLED", False)

# Feature 7: a question naming a whole-document artifact (guide, playbook,
# checklist, manual, SOP, onboarding, policy, journey, timeline) is upgraded to
# the synthesis profile (outline mode) even when otherwise classified as a
# plain lookup or procedure.
PLANNER_INTENT_EXPANSION_ENABLED = _bool("PLANNER_INTENT_EXPANSION_ENABLED", False)

# Feature 8: "every X", "all X", or "everything" upgrades a fact_lookup/
# procedural classification to the exhaustive profile (a full sweep). Does NOT
# override comparison, multi_hop, synthesis, or cross_document -- measured to
# regress both a synthesis and a cross_document question when it did.
EXHAUSTIVE_TRIGGER_ENABLED = _bool("EXHAUSTIVE_TRIGGER_ENABLED", False)

# ---------------------------------------------------------------------------
# Phase 4, Part A -- answer voice and answer formatting
#
# Both are PROMPT-SIDE ONLY: they add instructions to the answer model's system
# prompt and change nothing about retrieval, ranking, or how citations are
# built. Nothing in the codebase reformats or re-flows the model's output, so a
# formatting change cannot move a figure.
#
# Two flags rather than one because their risk profiles differ. Voice cannot
# change an answer's structure; formatting produces markdown blocks (tables,
# lists) that grounding remediation and the client's progressive reveal both
# have to handle -- so the riskier half is revertible without losing the
# harmless one.
#
# Both default ON, which is the D9 exception rather than the standing OFF rule:
# this phase's acceptance gate is that answers READ warm and are shaped to
# their content, and a flag defaulting OFF would ship a system that passes
# review by not doing the thing. Setting either false is the documented revert
# to Phase 3 answer text.
# ---------------------------------------------------------------------------

# Warm, human register (contractions, direct address, no corporate stock
# phrases). Wording only: it may not add a fact, a reassurance, or a caveat,
# and it may not soften the fixed refusal sentence.
ANSWER_VOICE_ENABLED = _bool("ANSWER_VOICE_ENABLED", True)

# Let the model pick the structure that fits the content: a table for
# comparisons and repeated-attribute lists, bullets for steps and
# enumerations, prose for explanations.
ANSWER_FORMAT_ENABLED = _bool("ANSWER_FORMAT_ENABLED", True)

# ---------------------------------------------------------------------------
# V2 -- relational store, authentication, and per-user isolation
#
# SQLite owns every relational entity (users, documents, sessions, messages);
# Chroma keeps ONLY vectors and their metadata. The two are joined on
# ``document_id``, which is stamped into each chunk's metadata at ingest.
# ---------------------------------------------------------------------------
SQLITE_PATH = PROJECT_ROOT / os.getenv("SQLITE_PATH", "smartdoc.db")

# Master switch for the whole multi-tenant layer. ON by default because Phase 1
# exists to make isolation real: with it OFF no scope is ever applied and the
# system reverts to known-good V1 single-user behaviour, which is the escape
# hatch, not the target state.
MULTI_USER_ENABLED = _bool("MULTI_USER_ENABLED", True)

# JWT signing. The default is a development-only placeholder; a real deployment
# MUST set JWT_SECRET. Startup refuses to run with the placeholder unless
# ALLOW_INSECURE_JWT_SECRET is set, because a known signing key means anyone can
# mint a token for any user_id and every isolation guarantee below collapses.
#
# `or` rather than `os.getenv(..., default)`: .env deliberately ships this key
# PRESENT but blank (a line for the user to fill in, not an absent one), and
# os.getenv only falls back to its default when the variable is missing
# entirely -- a present-but-empty value is returned as "". Without `or`, that
# resolves JWT_SECRET to "" silently: assert_signing_key_usable()'s check for
# the placeholder string would not match an empty string either, so the
# startup guard never fires and the app signs tokens with an empty secret with
# no warning at all -- quietly worse than the placeholder it was meant to be.
JWT_SECRET = os.getenv("JWT_SECRET") or "dev-only-insecure-secret-change-me"
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = _int("JWT_EXPIRE_MINUTES", 60 * 24 * 7)
ALLOW_INSECURE_JWT_SECRET = _bool("ALLOW_INSECURE_JWT_SECRET", True)

# bcrypt work factor. 12 is the common floor in 2026; raising it costs login
# latency linearly in 2**rounds.
BCRYPT_ROUNDS = _int("BCRYPT_ROUNDS", 12)

# Google OAuth (authlib). Absent credentials disable only the Google routes --
# email/password auth keeps working, so the app is runnable without them.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback"
)
# Where the OAuth callback sends the browser once a JWT has been issued. The
# token is appended as a query parameter for the Next.js client to store.
OAUTH_SUCCESS_REDIRECT = os.getenv("OAUTH_SUCCESS_REDIRECT", "")
# Signs the short-lived OAuth state cookie authlib needs. Independent of
# JWT_SECRET so rotating one does not invalidate the other. `or`, not a
# two-arg getenv default, for the same present-but-blank reason as JWT_SECRET
# above -- an empty string SessionMiddleware secret is a silent weakening, not
# a fallback to the placeholder.
SESSION_SECRET = os.getenv("SESSION_SECRET") or "dev-only-oauth-session-secret"

# Seeded development account. Everything indexed before V2 is adopted by this
# user rather than deleted, so the existing corpus stays usable for testing.
DEV_USER_EMAIL = os.getenv("DEV_USER_EMAIL", "dev@smartdoc.local")
DEV_USER_PASSWORD = os.getenv("DEV_USER_PASSWORD", "devpassword123")
DEV_USER_ID = os.getenv("DEV_USER_ID", "dev-user-0001")

# ---------------------------------------------------------------------------
# API limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 4000)

# ---------------------------------------------------------------------------
# Service wiring
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Browser origins allowed to call the API. An explicit allowlist replaces the
# Phase-1 ``*``: the wildcard was acceptable only because authorization is a
# bearer header rather than a cookie, but it also let any page on the machine
# read a signed-in user's documents by replaying a token it had scraped. Both
# spellings of localhost are listed because Next.js serves on ``localhost`` while
# the API's own defaults use ``127.0.0.1``, and the browser treats them as
# different origins.
CORS_ALLOW_ORIGINS = _csv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
