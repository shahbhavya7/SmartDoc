"""Measure one orchestration feature: flag OFF (baseline) vs flag ON.

Each new feature must be shown not to regress anything before the next is
built. This runs the labelled gold set twice -- once with the feature's flag
off, once on -- and prints a per-question comparison plus an aggregate table.

A REGRESSION is any question that was correct with the flag off and is wrong
with it on. Regressions are reported prominently and set a non-zero exit code,
so "build and measure one feature at a time, stop on regression" is enforced by
the tool rather than by remembering to look.

Usage:
    .venv/bin/python -m scripts.eval_feature --flag ROUTER_ENABLED
    .venv/bin/python -m scripts.eval_feature --flag PLANNER_ENABLED \
        --types synthesis,cross_document
    .venv/bin/python -m scripts.eval_feature --flag ROUTER_ENABLED --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import backend.config as config
from scripts.eval_rag import (
    GOLD_PATH,
    QuestionResult,
    build_fact_index,
    evaluate_question,
    summarise,
)

# Metrics worth comparing side by side. Latency is included because these
# features cost extra LLM calls and that trade-off should be visible.
COMPARE = [
    ("Answer correctness", "answer_correctness", "higher"),
    ("Retrieval precision", "retrieval_precision", "higher"),
    ("Retrieval recall", "retrieval_recall", "higher"),
    ("Context relevance", "context_relevance", "higher"),
    ("Context completeness", "context_completeness", "higher"),
    ("Faithfulness", "faithfulness", "higher"),
    ("Citation coverage", "citation_coverage", "higher"),
    ("False refusals", "false_refusals", "lower"),
    ("Median latency (ms)", "median_latency_ms", "lower"),
]


def _run_pass(entries: list[dict], fact_index) -> list[QuestionResult]:
    """Evaluate every question once, recording rather than raising failures."""
    results: list[QuestionResult] = []
    for i, entry in enumerate(entries, start=1):
        print(f"    [{i}/{len(entries)}] {entry['id']}", flush=True)
        try:
            results.append(evaluate_question(entry, fact_index))
        except Exception as exc:  # noqa: BLE001
            print(f"        ERROR: {type(exc).__name__}: {exc}", flush=True)
            results.append(
                QuestionResult(
                    id=entry["id"],
                    question=entry["question"],
                    expected_type=entry.get("type", ""),
                    expect_refusal=bool(entry.get("expect_refusal")),
                    notes=[f"EVALUATION ERROR: {type(exc).__name__}: {exc}"],
                )
            )
    return results


def _fmt(value, width: int = 8) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    if isinstance(value, float):
        return f"{value:>{width}.3f}"
    return f"{value:>{width}}"


def _delta(before, after, direction: str) -> str:
    if before is None or after is None:
        return "     -"
    diff = after - before
    if abs(diff) < 1e-9:
        return "     ="
    better = (diff > 0) if direction == "higher" else (diff < 0)
    mark = "+" if diff > 0 else ""
    return f"{mark}{diff:.3f} {'OK' if better else 'WORSE'}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flag", required=True, help="Config flag to toggle, e.g. ROUTER_ENABLED."
    )
    parser.add_argument("--data-dir", type=Path, default=config.PROJECT_ROOT / "data")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--types", default=None, help="Restrict to these query types.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not hasattr(config, args.flag):
        print(f"Unknown config flag: {args.flag}", file=sys.stderr)
        raise SystemExit(2)

    entries = json.loads(args.gold.read_text(encoding="utf-8"))["questions"]
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        entries = [e for e in entries if e.get("type") in wanted]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("No questions selected.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Feature flag : {args.flag}")
    print(f"Questions    : {len(entries)}")
    print("Indexing corpus fact locations ...")
    fact_index = build_fact_index(args.data_dir)
    print(f"  {fact_index.page_count()} pages indexed.\n")

    original = getattr(config, args.flag)
    started = time.time()
    try:
        print(f"=== PASS 1/2: {args.flag}=False (baseline) ===")
        setattr(config, args.flag, False)
        before = _run_pass(entries, fact_index)

        print(f"\n=== PASS 2/2: {args.flag}=True ===")
        setattr(config, args.flag, True)
        after = _run_pass(entries, fact_index)
    finally:
        setattr(config, args.flag, original)

    before_by_id = {r.id: r for r in before}
    after_by_id = {r.id: r for r in after}

    regressions: list[tuple[str, str, str]] = []
    improvements: list[str] = []
    for qid in before_by_id:
        b, a = before_by_id[qid], after_by_id.get(qid)
        if a is None:
            continue
        if b.correct and not a.correct:
            regressions.append((qid, b.answer[:90], a.answer[:90]))
        elif a.correct and not b.correct:
            improvements.append(qid)

    print("\n" + "=" * 100)
    print(f"PER-QUESTION  ({args.flag}: OFF -> ON)")
    print("=" * 100)
    print(f"{'id':28} {'correct':>16} {'prec':>14} {'ctxrel':>14} {'ms':>14}")
    print("-" * 100)
    for qid in before_by_id:
        b, a = before_by_id[qid], after_by_id.get(qid)
        if a is None:
            continue
        flag = ""
        if b.correct and not a.correct:
            flag = "  <== REGRESSION"
        elif a.correct and not b.correct:
            flag = "  <== improved"
        print(
            f"{qid:28} "
            f"{str(b.correct):>7}->{str(a.correct):<8}"
            f"{_fmt(b.precision, 6)}->{_fmt(a.precision, 6)}"
            f"{_fmt(b.context_relevance, 6)}->{_fmt(a.context_relevance, 6)}"
            f"{b.latency_ms:>7}->{a.latency_ms:<7}{flag}"
        )

    summary_before = summarise(before)
    summary_after = summarise(after)

    print("\n" + "=" * 100)
    print("AGGREGATE")
    print("=" * 100)
    print(f"{'metric':26} {'OFF':>10} {'ON':>10}   {'delta':<18}")
    print("-" * 100)
    for label, key, direction in COMPARE:
        b, a = summary_before.get(key), summary_after.get(key)
        print(f"{label:26} {_fmt(b, 10)} {_fmt(a, 10)}   {_delta(b, a, direction):<18}")

    print("\n" + "=" * 100)
    if regressions:
        print(f"RESULT: {len(regressions)} REGRESSION(S) -- do not ship this feature")
        for qid, before_answer, after_answer in regressions:
            print(f"\n  {qid}")
            print(f"    OFF (correct): {before_answer}")
            print(f"    ON  (wrong)  : {after_answer}")
    else:
        print("RESULT: no regressions")
    if improvements:
        print(f"\nImproved: {', '.join(improvements)}")
    print(f"\nCompleted in {time.time() - started:.0f}s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "flag": args.flag,
                    "summary_off": summary_before,
                    "summary_on": summary_after,
                    "regressions": [r[0] for r in regressions],
                    "improvements": improvements,
                    "results_off": [asdict(r) for r in before],
                    "results_on": [asdict(r) for r in after],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")

    raise SystemExit(1 if regressions else 0)


if __name__ == "__main__":
    main()
