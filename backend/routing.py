"""Document-level routing: decide WHICH documents a question is about.

Chunk retrieval previously ran flat against the whole corpus, so relevance was
only ever judged per chunk. A chunk from an unrelated document can clear a
per-chunk bar on loose vocabulary overlap -- and did: an off-topic PDF took a
quarter of the context on synthesis questions. Only a document-level decision
can exclude it, because the evidence that a document is off-topic is not visible
in any single one of its chunks.

    query -> score documents -> select documents -> chunk retrieval (filtered)

Scoring
-------
Documents are scored from a corpus-wide sample of chunk hits, fusing dense and
keyword rankings by RRF, then summing each document's **top few** chunk scores.
Summing *all* hits would rank by length: a 33-page manual accumulates more weak
matches than a 6-page register that answers the question exactly. Bounding the
sum at ``DOC_SCORE_TOP_CHUNKS`` makes the score reflect peak relevance with a
small reward for corroboration.

Selection
---------
Relative, not absolute: a document survives if its score is at least
``DOC_SCORE_DROP_RATIO`` of the leader's. Absolute thresholds do not transfer
across corpora, embedding models, or phrasings, whereas the ratio between the
best and second-best document is scale-free.

The top document is always kept, so routing can never return nothing. Multi-hop
and cross-document intents opt out of gating entirely -- their bridging evidence
lives in a *different* document by definition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import backend.config as config
from backend.vectorstore import (
    all_chunks,
    get_chunks_where,
    get_collection,
    query_collection,
)

logger = logging.getLogger("smartdoc.routing")


@dataclass
class DocumentScore:
    """A document's routing score and the evidence behind it."""

    source: str
    score: float
    doc_title: str = ""
    top_sections: list[str] = field(default_factory=list)
    chunk_hits: int = 0


@dataclass
class RoutingDecision:
    """Which documents chunk retrieval is allowed to draw from."""

    selected: list[str] = field(default_factory=list)
    scores: list[DocumentScore] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    gated: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "excluded": self.excluded,
            "gated": self.gated,
            "reason": self.reason,
            "scores": [
                {"source": s.source, "score": round(s.score, 5), "hits": s.chunk_hits}
                for s in self.scores
            ],
        }


def score_documents(
    queries: list[str],
    vectors: list[list[float]],
    keyword_query: str,
    collection_name: str | None = None,
    persist_directory=None,
) -> list[DocumentScore]:
    """Rank documents by peak chunk relevance to ``queries``.

    ``vectors`` are reused from the caller, so routing costs no extra embedding
    calls.
    """
    # Imported here to avoid a circular import at module load: retrieval imports
    # routing, and the keyword index lives in retrieval.
    from backend.retrieval import keyword_search, reciprocal_rank_fusion

    ranked_lists: list[list[str]] = []
    chunk_meta: dict[str, dict] = {}

    for vector in vectors:
        hits = query_collection(
            query_vector=vector,
            top_k=config.DOC_ROUTING_CANDIDATES,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        ranked_lists.append([h["id"] for h in hits])
        for hit in hits:
            chunk_meta.setdefault(hit["id"], hit["metadata"])

    if config.ENABLE_HYBRID:
        for text in {keyword_query, *queries}:
            hits = keyword_search(
                text, config.DOC_ROUTING_CANDIDATES, collection_name, persist_directory
            )
            ranked_lists.append([h["id"] for h in hits])
            for hit in hits:
                chunk_meta.setdefault(hit["id"], hit["metadata"])

    fused = reciprocal_rank_fusion(ranked_lists)

    per_document: dict[str, list[float]] = {}
    sections: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    for chunk_id, score in fused.items():
        meta = chunk_meta.get(chunk_id) or {}
        source = meta.get("source")
        if not source:
            continue
        per_document.setdefault(source, []).append(score)
        titles.setdefault(source, meta.get("doc_title", ""))
        section = meta.get("section")
        if section:
            sections.setdefault(source, []).append(section)

    results: list[DocumentScore] = []
    for source, scores in per_document.items():
        scores.sort(reverse=True)
        # Bounded sum: peak relevance plus a little credit for corroboration,
        # without letting document length dominate.
        results.append(
            DocumentScore(
                source=source,
                score=sum(scores[: config.DOC_SCORE_TOP_CHUNKS]),
                doc_title=titles.get(source, ""),
                top_sections=list(dict.fromkeys(sections.get(source, [])))[:5],
                chunk_hits=len(scores),
            )
        )

    results.sort(key=lambda d: d.score, reverse=True)
    return results


def select_documents(
    scores: list[DocumentScore],
    max_documents: int,
    gate: bool,
    drop_ratio: float | None = None,
) -> RoutingDecision:
    """Choose the documents chunk retrieval may draw from."""
    if not scores:
        return RoutingDecision(reason="no documents scored")

    ratio = config.DOC_SCORE_DROP_RATIO if drop_ratio is None else drop_ratio
    top_score = scores[0].score

    if not gate:
        return RoutingDecision(
            selected=[s.source for s in scores[:max_documents]],
            scores=scores,
            excluded=[],
            gated=False,
            reason=(
                "cross-document intent: routing observed but not enforced, "
                "because bridging evidence is expected in other documents"
            ),
        )

    selected: list[str] = []
    excluded: list[str] = []
    for index, doc in enumerate(scores):
        if index == 0:
            # The leader is always kept: routing must never return nothing.
            selected.append(doc.source)
            continue
        if len(selected) >= max_documents or doc.score < top_score * ratio:
            excluded.append(doc.source)
            continue
        selected.append(doc.source)

    return RoutingDecision(
        selected=selected,
        scores=scores,
        excluded=excluded,
        gated=True,
        reason=(
            f"kept documents scoring >= {ratio:.2f} x top score "
            f"({top_score:.4f}), max {max_documents}"
        ),
    )


def document_outline(
    source: str, collection_name: str | None = None, persist_directory=None
) -> list[dict]:
    """Return ``source``'s section outline in document order.

    Built from the ``section``/``page`` metadata already on every chunk, so it
    needs no extra parsing pass and stays consistent with what is indexed. A
    document-wide synthesis question is answered by covering the outline, not by
    whichever passages happen to embed closest to a vague question.
    """
    collection = get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    # Via the scoped helper, so an outline is built only from chunks the
    # signed-in user owns.
    entries: dict[str, dict] = {}
    for chunk in get_chunks_where(
        {"source": source}, collection=collection, include=["metadatas"]
    ):
        meta = chunk["metadata"] or {}
        section = (meta.get("section") or "").strip()
        if not section:
            continue
        page = int(meta.get("page", 0) or 0)
        existing = entries.get(section)
        if existing is None or page < existing["page"]:
            entries[section] = {
                "section": section,
                "page": page,
                "doc_title": meta.get("doc_title", ""),
                "source": source,
            }
    return sorted(entries.values(), key=lambda e: e["page"])


def document_chunks(
    sources: list[str], collection_name: str | None = None, persist_directory=None
) -> list[dict]:
    """Return every indexed chunk belonging to ``sources``, in document order.

    Used by exhaustive extraction, which must examine every section rather than
    the highest-scoring few. Reading the whole document back is only tractable
    because the candidate set is already narrowed to the routed documents.
    """
    if not sources:
        return []
    collection = get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    where = (
        {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
    )
    chunks = get_chunks_where(where, collection=collection)
    chunks.sort(
        key=lambda c: (
            c["metadata"].get("source", ""),
            int(c["metadata"].get("chunk_index", 0) or 0),
        )
    )
    return chunks


def corpus_documents(
    collection_name: str | None = None, persist_directory=None
) -> list[str]:
    """List every distinct source document currently indexed."""
    return sorted(
        {
            c["metadata"].get("source", "")
            for c in all_chunks(
                get_collection(
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                )
            )
        }
        - {""}
    )
