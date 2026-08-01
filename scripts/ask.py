"""Manual CLI entrypoint for the RAG query function.

Usage:
    .venv/bin/python -m scripts.ask "How many paid sick days do employees get?"
    .venv/bin/python -m scripts.ask "List all fault codes" --verbose
    .venv/bin/python -m scripts.ask "Who won the 2022 World Cup?" --json

Prints the grounded answer, its citations, the grounding verdict, and -- with
``--verbose`` -- the retrieval plan, document routing, and per-stage latency, so
the adaptive behaviour is inspectable from the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.rag import GenerationError, InvalidQuestionError, query
from backend.vectorstore import VectorStoreError

RULE = "=" * 74


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to ask (quote it).")
    parser.add_argument(
        "--top-k", type=int, default=None, help="Override the adaptive final k."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the retrieval plan, routing, and latency breakdown.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the full response as JSON."
    )
    args = parser.parse_args()

    try:
        result = query(args.question, top_k=args.top_k)
    except InvalidQuestionError as exc:
        print(f"Invalid question: {exc}", file=sys.stderr)
        sys.exit(2)
    except VectorStoreError as exc:
        print(f"Vector store error: {exc}", file=sys.stderr)
        sys.exit(3)
    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    diagnostics = result.diagnostics
    plan = diagnostics.get("plan", {})

    print(RULE)
    print(f"ANSWER   [intent: {result.query_type} | mode: {plan.get('mode')}]")
    print(RULE)
    print(result.answer)

    print()
    print(RULE)
    print("SOURCES")
    print(RULE)
    if not result.sources:
        print("(none -- no document supported an answer)")
    else:
        for i, src in enumerate(result.sources, start=1):
            pages = (
                f"pages {src.page}-{src.page_end}" if src.page_end else f"page {src.page}"
            )
            heading = f" | {src.section}" if src.section else ""
            print(f"[{i}] {src.source} ({pages}){heading}")
            print(f'    "{src.snippet}"')

    grounding = result.grounding
    if grounding.checked:
        print()
        print(RULE)
        print("GROUNDING")
        print(RULE)
        verdict = {True: "supported", False: "NOT FULLY SUPPORTED", None: "unknown"}[
            grounding.faithful
        ]
        print(f"verdict: {verdict}")
        if grounding.repaired:
            print(f"remediation: {grounding.repaired}")
            for claim in grounding.removed_claims:
                print(f"  - removed: {claim}")
        for claim in grounding.unsupported_claims:
            print(f"  ! unsupported: {claim}")
        for number in grounding.unverified_numbers:
            print(f"  ? figure not verbatim in context (may be derived): {number}")
        if grounding.note:
            print(f"note: {grounding.note}")

    if args.verbose:
        retrieval = diagnostics.get("retrieval", {})
        routing = diagnostics.get("routing") or {}
        print()
        print(RULE)
        print("RETRIEVAL PLAN")
        print(RULE)
        print(f"classified by : {plan.get('classified_by')}")
        if diagnostics.get("escalated_from"):
            print(f"escalated from: {diagnostics['escalated_from']}")
        print(f"final k       : {plan.get('final_k')} (candidates {plan.get('candidate_k')})")
        print(f"hybrid        : {diagnostics.get('hybrid')}   reranked: {diagnostics.get('reranked')}")
        if plan.get("subtopics"):
            print(f"subtopics     : {', '.join(plan['subtopics'])}")
        if plan.get("entities"):
            print(f"entities      : {', '.join(plan['entities'])}")
        for sub_query in plan.get("sub_queries", [])[:10]:
            print(f"  query: {sub_query}")

        print()
        print(f"documents kept    : {retrieval.get('documents_selected')}")
        print(f"documents excluded: {retrieval.get('documents_excluded')}")
        if routing.get("reason"):
            print(f"routing reason    : {routing['reason']}")
        print(f"stages            : {retrieval}")
        print(
            f"context           : {diagnostics.get('context_tokens')} tokens in "
            f"{diagnostics.get('context_blocks')} block(s); "
            f"{diagnostics.get('duplicates_removed')} duplicate(s) removed, "
            f"{diagnostics.get('adjacent_merges')} merge(s), "
            f"{diagnostics.get('dropped_units')} over budget"
        )
        print(f"latency (ms)      : {diagnostics.get('latency_ms')}")


if __name__ == "__main__":
    main()
