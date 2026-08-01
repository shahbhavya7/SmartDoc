"""Feature 1 -- additional document-routing signals (ROUTER_ENABLED).

The existing router (``backend.routing``) scores a document purely from the
similarity of its chunks: RRF over dense and BM25 rankings, bounded to each
document's top few chunks. That is a good signal but a narrow one. A document
whose *title* names the subject, or that the user just referred to as "this
document", or whose named entities appear verbatim in the question, carries
evidence that no individual chunk score expresses.

This module layers four extra signals over the scores the retriever already
produced. It does not re-retrieve, re-embed, or re-rank anything.

    title similarity     query terms overlapping the document title
    explicit reference   "this document", "the policy above", or the document
                         named outright in the question
    conversation focus   the document last discussed, when a caller supplies it
    entity overlap       capitalised/identifier-shaped terms shared with the
                         question

Bias toward precision, without dropping true positives
------------------------------------------------------
Two protections, because the failure mode of a router is silently deleting the
document that held the answer -- which is exactly what happened when routing was
first introduced here (multi-hop recall fell 0.62 -> 0.38):

1. The adjustment is **bounded** by ``ROUTER_SIGNAL_WEIGHT`` as a multiplier of
   the document's existing score, so these signals can reorder near-ties but
   cannot overturn strong retrieval evidence.
2. Any document scoring at least ``ROUTER_PROTECT_RATIO`` of the leader is
   **protected**: its score is never reduced. A document the retriever strongly
   supports cannot be demoted by a heuristic.

Signals therefore act mainly to *demote* weakly-supported off-topic documents
and *promote* near-tie on-topic ones -- the stated goal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import backend.config as config
from backend.routing import DocumentScore

# Words too common to carry routing evidence.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "many", "much",
    "must", "of", "on", "or", "our", "that", "the", "their", "there", "these",
    "this", "to", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "all", "every", "list", "give", "explain",
}

# Phrases that refer to a document already in play rather than naming one.
_EXPLICIT_REFERENCE_RE = re.compile(
    r"\b(this|that|the above|the same|the aforementioned|said)\s+"
    r"(document|policy|handbook|manual|guide|standard|sop|register|procedure)\b"
    r"|\b(the (?:document|policy|handbook|manual|guide) (?:above|we just|"
    r"i just|mentioned))\b",
    re.I,
)

# Identifier-shaped and proper-noun tokens: the terms most likely to name a
# specific entity that a document is about.
_ENTITY_RE = re.compile(
    r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b"
    r"|\b[A-Z]{1,4}-\d{1,4}\b"
    r"|\bTier\s?\d\b",
)


@dataclass
class RouterSignals:
    """The extra evidence gathered for one document."""

    source: str
    title_similarity: float = 0.0
    explicit_reference: float = 0.0
    conversation_focus: float = 0.0
    entity_overlap: float = 0.0

    @property
    def total(self) -> float:
        """Mean of the four signals, so each contributes at most a quarter."""
        return (
            self.title_similarity
            + self.explicit_reference
            + self.conversation_focus
            + self.entity_overlap
        ) / 4.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": round(self.title_similarity, 3),
            "reference": round(self.explicit_reference, 3),
            "focus": round(self.conversation_focus, 3),
            "entities": round(self.entity_overlap, 3),
            "total": round(self.total, 3),
        }


@dataclass
class RouterAdjustment:
    """What the extra signals did, for diagnostics and measurement."""

    applied: bool = False
    signals: list[RouterSignals] = field(default_factory=list)
    before: list[tuple[str, float]] = field(default_factory=list)
    after: list[tuple[str, float]] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    reordered: bool = False

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "reordered": self.reordered,
            "protected": self.protected,
            "signals": [s.to_dict() for s in self.signals],
            "before": [(s, round(v, 5)) for s, v in self.before],
            "after": [(s, round(v, 5)) for s, v in self.after],
        }


def _terms(text: str) -> set[str]:
    """Content words of ``text``, lowercased and stopword-filtered."""
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text)
        if w.lower() not in _STOP and len(w) > 2
    }


def _entities(text: str) -> set[str]:
    """Entity-shaped tokens (proper nouns, identifiers, tiers)."""
    found = {m.group(0).strip() for m in _ENTITY_RE.finditer(text)}
    return {e.lower() for e in found if e.lower() not in _STOP}


def _filename_terms(source: str) -> set[str]:
    """Terms derived from a filename, e.g. vendor_register.pdf -> vendor, register."""
    stem = re.sub(r"\.pdf$", "", source, flags=re.I)
    return _terms(stem.replace("_", " ").replace("-", " "))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gather_signals(
    question: str,
    scores: list[DocumentScore],
    conversation_focus: str | None = None,
) -> list[RouterSignals]:
    """Compute the four extra signals for each scored document.

    Args:
        question: the user's question, verbatim.
        scores: the existing document scores from ``backend.routing``.
        conversation_focus: filename of the document last discussed, if the
            caller tracks one. ``None`` disables that signal entirely.
    """
    question_terms = _terms(question)
    question_entities = _entities(question)
    references_something = bool(_EXPLICIT_REFERENCE_RE.search(question))

    out: list[RouterSignals] = []
    for doc in scores:
        title_terms = _terms(doc.doc_title) | _filename_terms(doc.source)
        signals = RouterSignals(source=doc.source)

        # 1. Title similarity -- overlap between the question and the document's
        #    title/filename. A title names what a document is ABOUT, which no
        #    single chunk necessarily states.
        signals.title_similarity = _jaccard(question_terms, title_terms)

        # 2. Explicit reference. A bare "this document" cannot name which one, so
        #    it only boosts the conversation focus; a question naming the
        #    document (or its title words) boosts that document directly.
        named = bool(title_terms) and title_terms <= question_terms
        if named:
            signals.explicit_reference = 1.0
        elif references_something and conversation_focus == doc.source:
            signals.explicit_reference = 1.0

        # 3. Conversation focus -- the document last discussed. Only ever a
        #    bonus, never a penalty on others, so a wrong focus cannot exclude
        #    the right document.
        if conversation_focus and conversation_focus == doc.source:
            signals.conversation_focus = 1.0

        # 4. Entity overlap -- named entities shared with the question, measured
        #    against the document's title and its best-matching section
        #    headings (already available; no extra retrieval).
        doc_entities = _entities(
            " ".join([doc.doc_title, doc.source.replace("_", " "), *doc.top_sections])
        )
        if question_entities:
            signals.entity_overlap = len(question_entities & doc_entities) / len(
                question_entities
            )

        out.append(signals)
    return out


def refine_scores(
    question: str,
    scores: list[DocumentScore],
    conversation_focus: str | None = None,
) -> tuple[list[DocumentScore], RouterAdjustment]:
    """Re-score documents using the extra signals, bounded and protected.

    Returns a NEW list; the input scores are not mutated, so a caller can always
    fall back to the retriever's own ordering.
    """
    adjustment = RouterAdjustment()
    if not scores:
        return scores, adjustment

    adjustment.before = [(d.source, d.score) for d in scores]
    signals = gather_signals(question, scores, conversation_focus)
    adjustment.signals = signals
    by_source = {s.source: s for s in signals}

    top_score = max(d.score for d in scores)
    protect_floor = top_score * config.ROUTER_PROTECT_RATIO
    weight = config.ROUTER_SIGNAL_WEIGHT

    refined: list[DocumentScore] = []
    for doc in scores:
        signal = by_source.get(doc.source)
        strength = signal.total if signal else 0.0

        # Centre the adjustment on zero: a document with no supporting signal is
        # damped, one with strong signals is boosted, and the magnitude is capped
        # at +/- weight of the original score.
        factor = 1.0 + weight * (2.0 * strength - 1.0)

        if doc.score >= protect_floor and factor < 1.0:
            # Protected: the retriever strongly supports this document, so the
            # heuristics may not demote it.
            factor = 1.0
            adjustment.protected.append(doc.source)

        refined.append(
            DocumentScore(
                source=doc.source,
                score=doc.score * factor,
                doc_title=doc.doc_title,
                top_sections=doc.top_sections,
                chunk_hits=doc.chunk_hits,
            )
        )

    refined.sort(key=lambda d: d.score, reverse=True)
    adjustment.after = [(d.source, d.score) for d in refined]
    adjustment.applied = True
    adjustment.reordered = [s for s, _ in adjustment.before] != [
        s for s, _ in adjustment.after
    ]
    return refined, adjustment
