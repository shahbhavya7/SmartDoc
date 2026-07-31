"""Manual CLI entrypoint for Phase 3's RAG query function.

Usage:
    .venv/bin/python scripts/ask.py "How many days of annual leave do
    full-time employees get?"

    .venv/bin/python scripts/ask.py "Who won the 2022 World Cup?" --top-k 6

Prints the grounded answer plus formatted citations (or "No sources" on the
refusal path). This is a thin wrapper around ``backend.rag.query`` for
manual testing -- it is not part of the FastAPI app (Phase 4) or the
Streamlit UI (Phases 5-6).
"""

from __future__ import annotations

import argparse
import sys

from backend.rag import GenerationError, InvalidQuestionError, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to ask (quote it).")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the number of chunks retrieved (default: config.TOP_K).",
    )
    args = parser.parse_args()

    try:
        result = query(args.question, top_k=args.top_k)
    except InvalidQuestionError as exc:
        print(f"Invalid question: {exc}", file=sys.stderr)
        sys.exit(2)
    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)
    if not result.sources:
        print("(none -- no document supported an answer)")
    else:
        for i, src in enumerate(result.sources, start=1):
            print(f"[{i}] {src.source} (page {src.page})")
            print(f"    \"{src.snippet}\"")


if __name__ == "__main__":
    main()
