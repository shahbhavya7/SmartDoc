"""Orchestration intent: which heavy features, if any, a question warrants.

This sits ON TOP of the existing query classifier. That classifier decides how
to RETRIEVE (seven intents, each with a retrieval mode); this one decides
whether the new orchestration features should run at all.

    point-lookup          one specific fact -> fast path, untouched
    document-wide         summarise/explain a whole document -> outline synthesis
    combination/workflow  build a procedure from several sections -> planner
    other                 anything else -> no orchestration

The mapping is derived from the existing ``QueryPlan`` rather than
re-classifying from scratch: a second independent classifier would be a second
thing to disagree with the first, and the retrieval intent already encodes most
of what is needed. A small amount of extra lexical evidence separates
"combination/workflow" (design a process, combine these, what should I do) from
a plain procedural lookup, because the existing classifier does not draw that
line -- it has no reason to.

Nothing here changes retrieval. A point-lookup keeps the exact path it has
today, which is the point: the heavy features must never touch simple questions.
"""

from __future__ import annotations

import re

from backend.query_analysis import (
    CROSS_DOCUMENT,
    EXHAUSTIVE,
    FACT_LOOKUP,
    MULTI_HOP,
    PROCEDURAL,
    SYNTHESIS,
    QueryPlan,
)

POINT_LOOKUP = "point-lookup"
DOCUMENT_WIDE = "document-wide"
COMBINATION = "combination/workflow"
OTHER = "other"

ORCHESTRATION_INTENTS = (POINT_LOOKUP, DOCUMENT_WIDE, COMBINATION, OTHER)

# Questions that ask for something to be BUILT from several parts, rather than
# looked up. "How do I submit an expense report?" is procedural but not a
# combination: the document already contains the ordered steps. "Design an
# onboarding workflow combining security and training requirements" is, because
# no single section contains the answer.
_COMBINATION_RE = re.compile(
    r"\b(design|devise|construct|build|create|propose|put together|come up with)\b"
    r".{0,40}\b(workflow|process|plan|procedure|checklist|approach|strategy|"
    r"programme|program|roadmap)\b"
    r"|\bcombin\w+\b"
    r"|\bend[- ]to[- ]end\b"
    r"|\bwhat should (?:i|we) do\b"
    r"|\bhow should (?:i|we) (?:approach|handle|structure|organise|organize)\b",
    re.I,
)

# Whole-document scope markers. Kept separate from the retrieval classifier's
# synthesis signal so this layer can be reasoned about (and disabled) alone.
_DOCUMENT_WIDE_RE = re.compile(
    r"\b(summar\w+|overview of|explain the (?:entire|whole|full)|"
    r"walk me through the (?:entire|whole)|everything (?:in|about)|"
    r"the (?:entire|whole|full) (?:document|policy|handbook|manual|guide|sop))\b",
    re.I,
)


def classify(plan: QueryPlan) -> str:
    """Return the orchestration intent for an already-analysed question.

    Args:
        plan: the existing ``QueryPlan``. Read-only -- this never mutates it.

    Returns:
        One of ``ORCHESTRATION_INTENTS``.
    """
    question = plan.question

    # Combination is checked first: it is the most specific, and a workflow
    # question often also trips the synthesis or procedural signals.
    if _COMBINATION_RE.search(question):
        return COMBINATION

    if plan.query_type == SYNTHESIS or _DOCUMENT_WIDE_RE.search(question):
        return DOCUMENT_WIDE

    if plan.query_type == FACT_LOOKUP:
        return POINT_LOOKUP

    # Multi-hop, comparison, exhaustive, procedural and cross-document questions
    # are already served well by their retrieval modes. They are deliberately
    # NOT routed into the heavy features: doing so would add cost and risk to
    # paths that measured correctly.
    if plan.query_type in (
        MULTI_HOP,
        PROCEDURAL,
        EXHAUSTIVE,
        CROSS_DOCUMENT,
    ):
        return OTHER

    return OTHER


def is_point_lookup(plan: QueryPlan) -> bool:
    """True if the question must keep the current fast path, unchanged."""
    return classify(plan) == POINT_LOOKUP
