"""`table-router` branch: a LangGraph state machine that classifies each
question as TABLE-RELATED or NORMAL and routes it to the configured backend,
wrapping the two EXISTING pipelines as nodes rather than reimplementing
either.

    CLASSIFY -> (table-related?) -> TABLE_NODE  -> END
                (normal?)        -> NORMAL_NODE -> END

Both pipeline nodes call code that already exists and is unmodified by this
module:

* the hybrid node calls ``backend.rag.query`` -- the SQL-backed exact
  counts/sums fast path (``backend.table_store``) already lives INSIDE that
  call, ahead of generation, so an aggregation/counting question answered via
  this node still gets it regardless of why the router sent it here.
* the colpali node calls ``colpali_experiment.answer.answer`` +
  ``to_rag_response``, the same adapter ``/ask`` already used for the manual
  toggle before this router existed.

Routing destinations are config, not literals baked into the graph:
``config.TABLE_ROUTE_BACKEND`` / ``config.NORMAL_ROUTE_BACKEND`` name which
backend ("hybrid" or "colpali") each branch calls, so an operator can point
both branches at the same backend, or swap them, by editing ``.env`` --
no code change, no redeploy of graph logic.

**Aggregation carve-out.** The classifier's table/normal split is
orthogonal to "is this an aggregation/counting question" -- a table-related
row lookup and a table-related COUNT are both table-related, but only the
count needs the SQL fast path, and that path only exists inside the hybrid
node. So aggregation detection is NOT a third graph branch: it is a
pre-classification override that forces the hybrid backend whenever
``backend.table_store`` would fire its SQL aggregate path for this question,
regardless of which way the table/normal classifier would otherwise have
routed it. This is what keeps the acceptance check true ("aggregation
queries continue to that path regardless of the table/normal
classification") even when TABLE_ROUTE_BACKEND is repointed at colpali --
colpali has no SQL fast path at all, so sending a count question there would
silently lose it.

Everything this graph touches -- auth, per-user isolation, session/memory
persistence, citation shape -- is unchanged: ``backend/main.py`` still owns
all of that around the single ``run`` call this module exposes, exactly as
it already did around ``backend.rag.query``/``colpali_experiment.answer``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

import backend.config as config
from backend.rag import RagResponse, query as hybrid_query
from backend.table_store import classify_table_relatedness
from backend.user_scope import user_scope

logger = logging.getLogger("smartdoc.router_graph")

# Same guard as backend/main.py's own ``_colpali_available`` (a separate,
# independently-computed check rather than an import from main -- main
# imports THIS module, so importing back would be circular). A missing
# experimental dependency (torch, colpali-engine) must degrade routing to
# ColPali the same way it already degrades main.py's manual `?backend=colpali`
# override: fall back to hybrid rather than crash the request inside
# ``_colpali_node``.
try:
    import colpali_experiment.answer  # noqa: F401

    _colpali_available = True
except ImportError:  # pragma: no cover - experimental dependency not installed
    _colpali_available = False

RouteBackend = Literal["hybrid", "colpali"]
RoutePath = Literal["table_colpali", "normal_hybrid", "sql_aggregation"]


class RouterState(TypedDict, total=False):
    """The graph's shared state. Every node reads/writes a subset of this."""

    user_id: str
    question: str
    conversation_context: str | None
    conversation_focus: str | None

    # Set by CLASSIFY.
    is_table_related: bool
    classification_reason: str
    forced_backend: RouteBackend | None  # aggregation override, or None

    # Set by whichever pipeline node ran.
    response: RagResponse
    backend_used: RouteBackend
    path: RoutePath

    # Wall-clock timing, in perf_counter seconds -- converted to latency_ms
    # only once, at the end of `run()`, so a resumed/retried node never
    # double-counts.
    started_at: float


@dataclass(frozen=True)
class RoutedAnswer:
    """What ``run()`` returns: the answer plus which path produced it and how
    long the whole router took, end to end.
    """

    response: RagResponse
    backend_used: RouteBackend
    path: RoutePath
    latency_ms: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _classify_node(state: RouterState) -> dict:
    """CLASSIFIER: table-related vs normal, plus the aggregation override.

    Aggregation detection reuses ``backend.table_store``'s own Decision 1
    resolution (via ``classify_table_relatedness``, which exposes the probe
    it built) rather than a second classifier -- a question whose probe
    resolved an ``aggregate`` kind (max/min/sum/count/row_count/filter) is
    forced to the hybrid backend, because that is the only backend with the
    SQL fast path wired in (see ``backend.rag.query``'s ``PARALLEL_SQL_LOOKUP_ENABLED``
    branch). This overrides TABLE_ROUTE_BACKEND for exactly these questions;
    every other table-related question still follows TABLE_ROUTE_BACKEND.
    """
    verdict = classify_table_relatedness(state["user_id"], state["question"])
    forced: RouteBackend | None = "hybrid" if verdict.probe.aggregate else None

    logger.info(
        "router classify: question=%r table_related=%s reason=%r "
        "aggregate=%r forced_backend=%s",
        state["question"],
        verdict.is_table_related,
        verdict.reason,
        verdict.probe.aggregate,
        forced,
    )

    return {
        "is_table_related": verdict.is_table_related,
        "classification_reason": verdict.reason,
        "forced_backend": forced,
    }


def _route(state: RouterState) -> str:
    """Conditional edge: which pipeline node to enter.

    Reads ``config.TABLE_ROUTE_BACKEND`` / ``config.NORMAL_ROUTE_BACKEND`` at
    CALL TIME (not import time), so changing either in ``.env`` takes effect
    on the next request with no code change and no graph rebuild.
    """
    if state.get("forced_backend"):
        backend = state["forced_backend"]
    elif state["is_table_related"]:
        backend = config.TABLE_ROUTE_BACKEND
    else:
        backend = config.NORMAL_ROUTE_BACKEND

    if backend == "colpali" and not _colpali_available:
        # Same fallback backend/main.py's manual `?backend=colpali` override
        # already applies -- a missing experimental dependency must never
        # crash a request inside _colpali_node; hybrid always works.
        logger.warning(
            "Routed to colpali but colpali_experiment is unavailable; "
            "falling back to hybrid."
        )
        backend = "hybrid"
    return "colpali_node" if backend == "colpali" else "hybrid_node"


def _hybrid_node(state: RouterState) -> dict:
    """Wraps ``backend.rag.query`` -- unmodified. Per-user scope is bound
    here, exactly as ``/ask`` already bound it around this same call before
    the router existed.
    """
    with user_scope(state["user_id"]):
        response = hybrid_query(
            state["question"],
            conversation_context=state.get("conversation_context"),
            conversation_focus=state.get("conversation_focus"),
        )

    sql_lookup = (response.diagnostics or {}).get("sql_lookup") or {}
    answered_by = sql_lookup.get("answered_by", "")
    path: RoutePath = "sql_aggregation" if answered_by == "sql_aggregation" else "normal_hybrid"
    if state.get("forced_backend") and path != "sql_aggregation":
        # The aggregate was detected at classification time but Decision 2
        # (backend.table_store's own strict trust check) discarded it after
        # the fact -- e.g. an ambiguous entity/column match. The question was
        # still correctly forced to hybrid (the only backend that could have
        # answered the aggregate at all); it just fell through to the
        # ordinary retrieval answer within that same node, same as the
        # flag-off/discarded-result behaviour `backend.rag.query` already has.
        path = "normal_hybrid"

    return {"response": response, "backend_used": "hybrid", "path": path}


def _colpali_node(state: RouterState) -> dict:
    """Wraps ``colpali_experiment.answer`` -- unmodified. Imported lazily,
    same guard ``backend/main.py`` already applies, so a missing optional
    dependency (torch, colpali-engine) degrades only this node, never the
    hybrid path or graph construction itself.
    """
    from colpali_experiment.answer import answer as colpali_answer
    from colpali_experiment.answer import to_rag_response

    visual_answer = colpali_answer(state["user_id"], state["question"])
    response = to_rag_response(visual_answer)
    return {"response": response, "backend_used": "colpali", "path": "table_colpali"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph():
    graph = StateGraph(RouterState)
    graph.add_node("classify", _classify_node)
    graph.add_node("hybrid_node", _hybrid_node)
    graph.add_node("colpali_node", _colpali_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route,
        {"hybrid_node": "hybrid_node", "colpali_node": "colpali_node"},
    )
    graph.add_edge("hybrid_node", END)
    graph.add_edge("colpali_node", END)
    return graph.compile()


# Built once at import: the graph's structure (nodes/edges) is fixed; only
# the routing DESTINATION each edge resolves to at call time depends on
# config, which `_route` reads fresh on every invocation.
_GRAPH = _build_graph()


def run(
    user_id: str,
    question: str,
    *,
    conversation_context: str | None = None,
    conversation_focus: str | None = None,
) -> RoutedAnswer:
    """Classify ``question`` and answer it via the routed backend.

    This is the ONLY entry point ``backend/main.py`` needs: it returns the
    same ``RagResponse`` either pipeline already produced, plus which path
    answered and the router's own end-to-end latency -- measured from here
    (before the classifier runs) to here (after the chosen pipeline node
    returns), so it covers classification + retrieval + generation +
    verification, exactly the span the brief asks for.
    """
    started = time.perf_counter()
    initial_state: RouterState = {
        "user_id": user_id,
        "question": question,
        "conversation_context": conversation_context,
        "conversation_focus": conversation_focus,
    }
    final_state = _GRAPH.invoke(initial_state)
    latency_ms = round((time.perf_counter() - started) * 1000)

    logger.info(
        "router done: question=%r path=%s backend=%s latency_ms=%d",
        question,
        final_state["path"],
        final_state["backend_used"],
        latency_ms,
    )

    return RoutedAnswer(
        response=final_state["response"],
        backend_used=final_state["backend_used"],
        path=final_state["path"],
        latency_ms=latency_ms,
    )
