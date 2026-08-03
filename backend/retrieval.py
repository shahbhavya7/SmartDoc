"""Retrieval: routing, hybrid search, fusion, intent modes, reranking, expansion.

Pipeline for one question (parameters come from the ``QueryPlan``):

1. **Multi-query fan-out.** Every sub-query and planned subtopic is searched
   independently. This is what makes comparison and synthesis work: a single
   embedding of "how does leave differ between Standard and Executive bands?"
   lands between the two passages and the nearer one takes every slot, whereas
   separate queries retrieve both.
2. **Document routing.** A document-level pass decides which documents the
   question is about, before any chunk is selected (see ``backend.routing``).
3. **Hybrid retrieval.** Dense vector search *and* BM25. Dense generalises
   across paraphrase; BM25 nails rare exact tokens ("E-07", "AES-256", "Tier 3")
   where embeddings are weakest because such tokens carry little distributional
   signal.
4. **Reciprocal Rank Fusion.** Result lists merge by ``1/(K + rank)`` rather
   than raw score. Cosine distances and BM25 scores are on incomparable scales,
   so weighted-sum fusion needs per-corpus constants that do not transfer; RRF
   needs only ranks.
5. **Mode-specific selection.** ``focused``, ``per_entity``, ``multi_hop``,
   ``outline``, ``sweep`` or ``broad`` -- see ``backend.query_analysis``.
6. **Reranking.** Relevance judgement by ``UTILITY_MODEL``, scoring each
   candidate 0-3. Fusion is recall-oriented and deliberately over-retrieves.
7. **Expansion.** Children are replaced by their parents (deduplicated), and
   neighbours are pulled in for ordered/exhaustive intents.

Every stage degrades gracefully: a failed rerank keeps fusion order, a missing
parent falls back to the child text, and a BM25 failure leaves dense results
intact.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field

import openai
from rank_bm25 import BM25Okapi

import backend.config as config
from backend.query_analysis import (
    MODE_OUTLINE,
    MODE_PER_ENTITY,
    MODE_SWEEP,
    QueryPlan,
)
from backend.routing import (
    RoutingDecision,
    document_chunks,
    document_outline,
    score_documents,
    select_documents,
)
from backend.user_scope import current_user_id
from backend.vectorstore import (
    all_chunks,
    get_chunks_by_ids,
    get_collection,
    get_parents,
    openai_embed_fn,
    query_collection,
)

logger = logging.getLogger("smartdoc.retrieval")

# Truncation applied to each candidate before it is shown to the reranker.
RERANK_SNIPPET_CHARS = 320

# Fallback minimum rerank score when a profile does not specify one.
RERANK_MIN_SCORE = 1

# A neighbour inherits this fraction of the score of the unit that pulled it in.
# Below 1.0 so a scored unit always outranks its own neighbour, but well above 0
# so the neighbour is not the first thing the token budget discards.
NEIGHBOUR_SCORE_DISCOUNT = 0.75


def _score_of(unit: "RetrievedUnit") -> float:
    """Effective relevance score, treating an unscored unit as zero."""
    return unit.rerank_score if unit.rerank_score is not None else 0.0


@dataclass
class RetrievedUnit:
    """One unit of context, after fusion/reranking/expansion."""

    id: str
    text: str
    metadata: dict
    fused_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None
    expanded_from: str = ""  # parent | neighbour | merged | reserve | ""

    # The text of the child chunk that actually matched, preserved when a unit
    # is expanded to its parent. Citation snippets are cut from this: the child
    # was selected by embedding similarity, so it is the semantically matched
    # span REGARDLESS of the question's language. Choosing the snippet by
    # lexical overlap instead silently degrades to head-of-section for any
    # non-English question.
    matched_text: str = ""

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0) or 0)


@dataclass
class RetrievalResult:
    """Retrieved context plus diagnostics for evaluation and debugging."""

    units: list[RetrievedUnit] = field(default_factory=list)
    plan: QueryPlan | None = None
    candidates_considered: int = 0
    reranked: bool = False
    hybrid: bool = False
    routing: RoutingDecision | None = None
    outline: list[dict] = field(default_factory=list)
    stages: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword index
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_bm25_lock = threading.Lock()
_bm25_cache: dict[tuple[str, int, str], tuple[BM25Okapi, list[dict]]] = {}


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that preserves hyphenated identifiers.

    "E-01" must survive as one token: splitting it into "e" and "01" makes the
    rare identifier indistinguishable from noise, defeating the whole reason for
    having a keyword index alongside the dense one.
    """
    return _TOKEN_RE.findall(text.lower())


def _keyword_index(collection_name: str | None = None, persist_directory=None):
    """Build (or reuse) the BM25 index over the active user's indexed chunks.

    Cached on ``(collection, chunk count, user)`` so it rebuilds automatically
    after an upload changes the corpus, without a manual invalidation step.

    The user is part of the key, not an afterthought: the index is a materialised
    copy of the corpus, so one shared across tenants would return another user's
    passages as keyword hits no metadata filter could catch. ``all_chunks``
    already restricts the source rows to the active user; the key makes sure the
    right index is the one that gets reused.
    """
    collection = get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    count = collection.count()
    key = (collection.name, count, current_user_id() or "")
    with _bm25_lock:
        cached = _bm25_cache.get(key)
        if cached:
            return cached
        chunks = all_chunks(collection)
        if not chunks:
            return None
        index = BM25Okapi([_tokenize(c["document"]) for c in chunks])
        # Evict only indexes built against a stale corpus version; other users'
        # current indexes stay warm, so one upload does not force every signed-in
        # user to rebuild on their next query.
        for stale in [k for k in _bm25_cache if k[0] == collection.name and k[1] != count]:
            del _bm25_cache[stale]
        _bm25_cache[key] = (index, chunks)
        return _bm25_cache[key]


def keyword_search(
    query: str, k: int, collection_name: str | None = None, persist_directory=None
) -> list[dict]:
    """Return the top-``k`` BM25 matches for ``query``."""
    built = _keyword_index(collection_name, persist_directory)
    if not built:
        return []
    index, chunks = built
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out: list[dict] = []
    for position in ranked[:k]:
        if scores[position] <= 0:
            break
        chunk = chunks[position]
        out.append(
            {
                "id": chunk["id"],
                "document": chunk["document"],
                "metadata": chunk["metadata"],
                "score": float(scores[position]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Fusion and diversity
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int | None = None
) -> dict[str, float]:
    """Fuse ranked id lists into ``{id: score}`` by RRF."""
    k = k if k is not None else config.RRF_K
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def apply_diversity(
    units: list[RetrievedUnit], max_per_source: int
) -> list[RetrievedUnit]:
    """Cap how many units any single document contributes, preserving order."""
    kept: list[RetrievedUnit] = []
    per_source: dict[str, int] = {}
    for unit in units:
        count = per_source.get(unit.source, 0)
        if count >= max_per_source:
            continue
        per_source[unit.source] = count + 1
        kept.append(unit)
    return kept


def _diversify_by_section(
    units: list[RetrievedUnit], max_per_section: int
) -> list[RetrievedUnit]:
    """Cap units per (source, section), preserving order.

    Document-wide synthesis needs BREADTH of sections, not depth in the one
    section that matched best. Capping per section converts a ranked list of
    chunks into coverage of the document's outline.
    """
    kept: list[RetrievedUnit] = []
    counts: dict[tuple[str, str], int] = {}
    for unit in units:
        key = (unit.source, unit.metadata.get("section", ""))
        if counts.get(key, 0) >= max_per_section:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(unit)
    return kept


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

_RERANK_PROMPT = """You score how useful each passage is for answering a \
question. Reply with JSON only.

Scale:
3 = contains a direct answer, an essential part of a multi-part answer, OR an \
intermediate fact the answer depends on
2 = clearly relevant supporting detail
1 = same topic, might contribute
0 = irrelevant to the question

Critical: a passage can be essential WITHOUT mentioning the subject of the \
question. Score a passage 3 when it supplies:
- a general RULE that applies to the category the question's subject belongs to \
(e.g. the question asks about payroll records; the passage states the rule for \
"Restricted data"), or
- a CLASSIFICATION, tier, or category assignment linking the subject to such a \
rule, or
- a DEFINITION the answer needs.
These bridging passages are exactly what multi-step questions require, and they \
routinely share no vocabulary with the question at all.

Judge each passage independently. Do not reward a passage merely for sharing \
vocabulary with the question. For questions asking for ALL items of a kind, \
score every passage contributing any item as 3.

Reply exactly: {"scores": [{"i": <passage number>, "s": <0-3>}, ...]}"""


def _rerank_batch(question: str, units: list[RetrievedUnit]) -> int:
    """Score one batch of units in place; return how many scores were applied."""
    listing = "\n\n".join(
        f"[{i}] {unit.text[:RERANK_SNIPPET_CHARS]}"
        for i, unit in enumerate(units, start=1)
    )
    try:
        from backend.vectorstore import _shared_openai

        completion = _shared_openai().chat.completions.create(
            model=config.UTILITY_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _RERANK_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nPassages:\n{listing}",
                },
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
        scores = payload.get("scores") or []
    except (openai.OpenAIError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Rerank batch failed, keeping fusion order: %s", exc)
        return 0

    applied = 0
    for entry in scores:
        try:
            index = int(entry["i"]) - 1
            score = float(entry["s"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(units):
            units[index].rerank_score = score
            applied += 1
    return applied


def llm_rerank(question: str, units: list[RetrievedUnit]) -> bool:
    """Score ``units`` in place against ``question``, batching as needed.

    Batching exists for exhaustive sweeps, which deliberately submit every
    section of a document rather than a top-k window. Truncating the candidate
    list instead would reintroduce the "stopped after the highest-scoring
    chunks" failure that sweeps exist to fix.
    """
    if not units:
        return False

    batch_size = max(5, config.RERANK_BATCH_SIZE)
    applied = 0
    for start in range(0, len(units), batch_size):
        applied += _rerank_batch(question, units[start : start + batch_size])
    return applied > 0


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def expand_to_parents(units: list[RetrievedUnit]) -> list[RetrievedUnit]:
    """Replace children by their parents, deduplicating shared parents.

    Several retrieved children frequently belong to one section. Emitting the
    parent once instead of three overlapping children both restores the full
    section and removes duplicated overlap text.
    """
    parent_ids = [u.metadata.get("parent_id") for u in units if u.metadata.get("parent_id")]
    records = get_parents(dict.fromkeys(parent_ids))

    out: list[RetrievedUnit] = []
    seen_parents: set[str] = set()
    for unit in units:
        parent_id = unit.metadata.get("parent_id")
        record = records.get(parent_id) if parent_id else None
        if not record:
            out.append(unit)
            continue
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        metadata = dict(unit.metadata)
        metadata["page"] = record.get("page", unit.metadata.get("page"))
        metadata["page_end"] = record.get("page_end", metadata.get("page_end"))
        metadata["section"] = record.get("section", metadata.get("section", ""))
        out.append(
            RetrievedUnit(
                id=parent_id,
                text=record["text"],
                metadata=metadata,
                fused_score=unit.fused_score,
                rerank_score=unit.rerank_score,
                dense_rank=unit.dense_rank,
                keyword_rank=unit.keyword_rank,
                expanded_from="parent",
                matched_text=unit.matched_text or unit.text,
            )
        )
    return out


def expand_neighbours(
    units: list[RetrievedUnit],
    collection_name: str | None = None,
    persist_directory=None,
) -> list[RetrievedUnit]:
    """Add the immediately adjacent chunks of each retrieved child.

    Ordered procedures and long tables routinely straddle a chunk boundary;
    "Step 4" is useless without "Step 5" on the other side of the split.
    """
    # Track which unit pulled in each neighbour, so the neighbour can inherit a
    # score. Neighbours are added AFTER reranking and are never scored
    # themselves; leaving them at zero sorts them below everything and the
    # context token budget drops them first. Observed: the vendor table
    # answering "list every Tier 3 vendor" was retrieved as a neighbour of a
    # chunk scored 3.0, then discarded before reaching the prompt -- so the
    # answer correctly said the names were not in the context. Expansion that is
    # silently undone by budgeting is worse than no expansion, because it looks
    # like it worked.
    origin: dict[str, RetrievedUnit] = {}
    for unit in units:
        for key in ("prev_id", "next_id"):
            neighbour = unit.metadata.get(key)
            if neighbour:
                current = origin.get(neighbour)
                if current is None or _score_of(unit) > _score_of(current):
                    origin[neighbour] = unit

    for existing in units:
        origin.pop(existing.id, None)
    if not origin:
        return units

    collection = get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    # get(ids=...) cannot take a where clause, so ownership is applied to the
    # results inside get_chunks_by_ids. Neighbour ids are user-namespaced and
    # come from chunks already retrieved under this scope, so this is a second
    # barrier rather than the only one.
    extra: list[RetrievedUnit] = []
    for neighbour_chunk in get_chunks_by_ids(sorted(origin), collection=collection):
        cid = neighbour_chunk["id"]
        doc = neighbour_chunk["document"]
        meta = neighbour_chunk["metadata"]
        parent_unit = origin.get(cid)
        inherited = _score_of(parent_unit) * NEIGHBOUR_SCORE_DISCOUNT if parent_unit else 0.0
        extra.append(
            RetrievedUnit(
                id=cid,
                text=doc,
                metadata=meta,
                expanded_from="neighbour",
                rerank_score=inherited or None,
                fused_score=parent_unit.fused_score if parent_unit else 0.0,
            )
        )
    return units + extra


# ---------------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------------


def _source_filter(allowed: list[str] | None) -> dict | None:
    """Build a Chroma metadata filter restricting retrieval to ``allowed``."""
    if not allowed:
        return None
    if len(allowed) == 1:
        return {"source": allowed[0]}
    return {"source": {"$in": list(allowed)}}


def _add_hits(
    pool: dict[str, RetrievedUnit], hits: list[dict], kind: str
) -> list[str]:
    """Merge hits into ``pool``, tracking best rank per retriever."""
    ids: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        ids.append(hit["id"])
        unit = pool.get(hit["id"])
        if unit is None:
            unit = RetrievedUnit(
                id=hit["id"], text=hit["document"], metadata=hit["metadata"]
            )
            pool[hit["id"]] = unit
        if kind == "dense" and (unit.dense_rank is None or rank < unit.dense_rank):
            unit.dense_rank = rank
        if kind == "keyword" and (unit.keyword_rank is None or rank < unit.keyword_rank):
            unit.keyword_rank = rank
    return ids


def _search_pool(
    queries: list[str],
    vectors: list[list[float]],
    keyword_query: str,
    candidate_k: int,
    where: dict | None,
    collection_name: str | None,
    persist_directory,
) -> tuple[dict[str, RetrievedUnit], list[list[str]]]:
    """Fan out every query over dense + keyword search and fuse.

    ``per_query_dense`` preserves which query found what -- needed by per-entity
    mode to guarantee each compared entity its own slots.
    """
    pool: dict[str, RetrievedUnit] = {}
    ranked_lists: list[list[str]] = []
    per_query_dense: list[list[str]] = []

    for vector in vectors:
        dense = query_collection(
            query_vector=vector,
            top_k=candidate_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
            where=where,
        )
        ids = _add_hits(pool, dense, "dense")
        ranked_lists.append(ids)
        per_query_dense.append(ids)

    if config.ENABLE_HYBRID:
        allowed = None
        if where:
            source = where.get("source")
            allowed = {source} if isinstance(source, str) else set(source.get("$in", []))
        for query_text in {keyword_query, *queries}:
            # BM25 runs over the whole index, so document gating is applied
            # after the fact rather than inside the scorer.
            lexical = keyword_search(
                query_text, candidate_k * 2, collection_name, persist_directory
            )
            if allowed:
                lexical = [h for h in lexical if h["metadata"].get("source") in allowed]
            ranked_lists.append(_add_hits(pool, lexical[:candidate_k], "keyword"))

    fused = reciprocal_rank_fusion(ranked_lists)
    for doc_id, score in fused.items():
        if doc_id in pool:
            pool[doc_id].fused_score = score

    return pool, per_query_dense


def _reserve_cross_document(
    pool: dict[str, RetrievedUnit],
    vectors: list[list[float]],
    allowed: list[str],
    collection_name: str | None,
    persist_directory,
) -> int:
    """Add a few top candidates from OUTSIDE the routed documents.

    Routing should bias retrieval, not blind it. When a question is
    misclassified as a simple lookup, hard gating deletes the bridging document:
    multi-hop recall fell 0.62 -> 0.38 when gating was introduced, entirely on
    questions the classifier had labelled fact_lookup. Reserving a few slots
    lets the reranker still see the bridge; the reranker then decides, so the
    cost is a little precision rather than a wrong answer.
    """
    slots = config.CROSS_DOC_RESERVE_SLOTS
    if slots <= 0 or not allowed:
        return 0

    added = 0
    for vector in vectors[:2]:  # the question and, at most, the first sub-query
        hits = query_collection(
            query_vector=vector,
            top_k=slots * 3,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        for hit in hits:
            if added >= slots:
                break
            if hit["metadata"].get("source") in allowed or hit["id"] in pool:
                continue
            pool[hit["id"]] = RetrievedUnit(
                id=hit["id"],
                text=hit["document"],
                metadata=hit["metadata"],
                expanded_from="reserve",
            )
            added += 1
    return added


def _sweep_pool(
    allowed: list[str],
    keyword_query: str,
    collection_name: str | None,
    persist_directory,
) -> tuple[dict[str, RetrievedUnit], bool]:
    """Load EVERY chunk of the routed documents as a candidate.

    Exhaustive extraction cannot use top-k: items of the same kind appear in
    several places and similarity ranking stops at the densest one. Reading every
    section and letting the reranker judge each is only affordable because
    routing has already narrowed the corpus to one or two documents.

    Above ``SWEEP_MAX_CANDIDATES`` the sweep degrades to a keyword pre-filter so a
    very large document stays tractable; the truncation is reported rather than
    hidden, because a silent cut is indistinguishable from "there was nothing
    else".
    """
    chunks = document_chunks(allowed, collection_name, persist_directory)
    truncated = False

    if len(chunks) > config.SWEEP_MAX_CANDIDATES:
        truncated = True
        allowed_set = set(allowed)
        scored = keyword_search(
            keyword_query,
            config.SWEEP_MAX_CANDIDATES * 2,
            collection_name,
            persist_directory,
        )
        keep = {
            hit["id"]
            for hit in scored
            if hit["metadata"].get("source") in allowed_set
        }
        filtered = [c for c in chunks if c["id"] in keep][: config.SWEEP_MAX_CANDIDATES]
        chunks = filtered or chunks[: config.SWEEP_MAX_CANDIDATES]

    pool = {
        chunk["id"]: RetrievedUnit(
            id=chunk["id"], text=chunk["document"], metadata=chunk["metadata"]
        )
        for chunk in chunks
    }
    return pool, truncated


def _per_entity_window(
    pool: dict[str, RetrievedUnit],
    per_query_dense: list[list[str]],
    slots_per_entity: int,
) -> list[RetrievedUnit]:
    """Guarantee each compared entity its own retrieval slots.

    A comparison fails when one entity's section outscores the other's and takes
    every slot, so the answer covers one side and implies symmetry. Round-robin
    selection over the per-entity query results makes each side's evidence
    structurally guaranteed rather than dependent on which section embeds closer.
    """
    entity_lists = per_query_dense[1:]  # index 0 is the original question
    if not entity_lists:
        return []

    picked: list[RetrievedUnit] = []
    seen: set[str] = set()
    for depth in range(slots_per_entity):
        for ids in entity_lists:
            if depth >= len(ids):
                continue
            unit = pool.get(ids[depth])
            if unit and unit.id not in seen:
                seen.add(unit.id)
                picked.append(unit)
    return picked


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def retrieve(
    plan: QueryPlan,
    collection_name: str | None = None,
    persist_directory=None,
    embed_fn=None,
    conversation_focus: str | None = None,
) -> RetrievalResult:
    """Run intent-aware, hierarchically-routed retrieval for ``plan``.

    Args:
        conversation_focus: filename of the document last discussed, used only
            by the optional router signals (Feature 1). ``None`` -- the default
            -- leaves that signal inactive, so stateless callers are unaffected.
    """
    profile = plan.profile
    queries = plan.sub_queries or [plan.question]

    # A keyword query enriched with extracted rare tokens; repeating them raises
    # their BM25 term frequency, which is the intended effect for
    # identifier-shaped terms.
    keyword_query = " ".join([plan.question] + plan.keywords)

    embed = embed_fn or openai_embed_fn()
    vectors = embed(queries)  # one batched call covers every sub-query

    # ---------------- 1. hierarchical document routing ----------------
    routing: RoutingDecision | None = None
    allowed: list[str] | None = None
    router_adjustment = None
    if config.ENABLE_DOC_ROUTING:
        scores = score_documents(
            queries, vectors, keyword_query, collection_name, persist_directory
        )
        # Feature 1 (ROUTER_ENABLED, default OFF): layer extra routing signals
        # over the scores the retriever produced. Inert when the flag is off --
        # `scores` is passed through untouched, so behaviour is identical.
        if config.ROUTER_ENABLED:
            from backend.doc_router import refine_scores

            scores, router_adjustment = refine_scores(
                plan.question, scores, conversation_focus=conversation_focus
            )
        routing = select_documents(
            scores,
            profile.max_documents,
            gate=profile.restrict_documents,
            drop_ratio=profile.doc_gate_ratio,
        )
        if routing.gated and routing.selected:
            allowed = routing.selected

    where = _source_filter(allowed)
    mode = profile.mode
    outline: list[dict] = []
    truncated_sweep = False
    reserved = 0

    # ---------------- 2. mode-specific candidate pool ----------------
    if mode == MODE_SWEEP and allowed:
        pool, truncated_sweep = _sweep_pool(
            allowed, keyword_query, collection_name, persist_directory
        )
        per_query_dense: list[list[str]] = []
        ordered = list(pool.values())
    else:
        pool, per_query_dense = _search_pool(
            queries,
            vectors,
            keyword_query,
            profile.candidate_k,
            where,
            collection_name,
            persist_directory,
        )
        if allowed:
            reserved = _reserve_cross_document(
                pool, vectors, allowed, collection_name, persist_directory
            )
        ordered = sorted(pool.values(), key=lambda u: u.fused_score, reverse=True)

    candidates_considered = len(pool)

    if mode == MODE_OUTLINE and allowed:
        for source in allowed:
            outline.extend(document_outline(source, collection_name, persist_directory))

    # ---------------- 3. mode-specific selection ----------------
    if mode == MODE_PER_ENTITY and plan.entities and per_query_dense:
        slots = max(2, profile.final_k // max(1, len(plan.entities)))
        guaranteed = _per_entity_window(pool, per_query_dense, slots)
        guaranteed_ids = {u.id for u in guaranteed}
        candidates = guaranteed + [u for u in ordered if u.id not in guaranteed_ids]
        candidates = apply_diversity(candidates, profile.max_per_source)
    elif mode == MODE_OUTLINE:
        # Breadth across sections beats depth within one section.
        candidates = _diversify_by_section(ordered, max_per_section=2)
    elif mode == MODE_SWEEP:
        candidates = ordered
    else:
        candidates = apply_diversity(ordered, profile.max_per_source)

    # ---------------- 4. reranking ----------------
    if mode == MODE_SWEEP:
        window = candidates  # every candidate is judged; nothing dropped first
    else:
        window = candidates[: max(profile.final_k * 4, profile.final_k + 10)]

    reranked = False
    if config.ENABLE_RERANK and len(window) > profile.final_k:
        reranked = llm_rerank(plan.question, window)

    min_score = profile.min_rerank_score
    if reranked:
        survivors = [
            u for u in window if u.rerank_score is None or u.rerank_score >= min_score
        ]
        if mode == MODE_SWEEP:
            # Keep EVERY passing section. The context token budget trims if
            # necessary and reports what it dropped -- unlike a silent top-k cut.
            survivors.sort(
                key=lambda u: (u.source, int(u.metadata.get("chunk_index", 0) or 0))
            )
            selected = survivors or window[: profile.final_k]
        else:
            survivors.sort(
                key=lambda u: (
                    u.rerank_score if u.rerank_score is not None else 0.0,
                    u.fused_score,
                ),
                reverse=True,
            )
            selected = survivors[: profile.final_k] or window[: profile.final_k]
    else:
        selected = window[: profile.final_k]

    # ---------------- 5. expansion ----------------
    if profile.expand_neighbours:
        selected = expand_neighbours(selected, collection_name, persist_directory)
    if config.ENABLE_PARENT_EXPANSION:
        selected = expand_to_parents(selected)

    return RetrievalResult(
        units=selected,
        plan=plan,
        candidates_considered=candidates_considered,
        reranked=reranked,
        hybrid=config.ENABLE_HYBRID,
        routing=routing,
        outline=outline,
        stages={
            "mode": mode,
            "queries": len(queries),
            "pool": candidates_considered,
            "reserved_cross_document": reserved,
            "after_selection": len(candidates),
            "rerank_window": len(window),
            "selected": len(selected),
            "documents_selected": allowed or "unrestricted",
            "documents_excluded": routing.excluded if routing else [],
            "sweep_truncated": truncated_sweep,
            "router_signals": (
                router_adjustment.to_dict() if router_adjustment else None
            ),
        },
    )
