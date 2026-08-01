"""Evaluation harness: retrieval, context, and answer quality metrics.

An earlier eval measured one thing -- whether the expected source/page appeared
in top-k for three questions -- and reported a tie across every configuration,
which is what an under-powered metric on an under-powered corpus looks like.
This measures the pipeline stage by stage, so a regression can be attributed
rather than guessed at.

Metrics
-------
Retrieval
  precision@k   retrieved units whose (source, page) is gold / units retrieved
  recall        gold (source, page) pairs retrieved / gold pairs that exist
  MRR           1 / rank of the first gold unit
  source recall gold documents retrieved / gold documents

Context
  relevance     share of assembled context TOKENS drawn from gold pages.
                Token-weighted rather than block-counted, because one large
                irrelevant block does more damage than three small ones.
  completeness  required fact strings present in the assembled context. This is
                the ceiling on answer quality: a fact absent here cannot be
                answered correctly except by guessing.

Answer
  correctness   required tokens present in the answer (structural, objective)
  faithfulness  LLM entailment judge, from the live pipeline
  hallucination answers with unsupported claims or unverified figures
  citation      gold documents cited / gold documents
  refusal       correct refusals out-of-scope, false refusals on answerable

Gold ``(source, page)`` pairs are located by searching the parsed corpus for each
required fact string, so labels survive re-chunking. A fact that cannot be
located is reported as gold-set drift rather than silently scoring zero.

Usage:
    .venv/bin/python -m scripts.eval_rag
    .venv/bin/python -m scripts.eval_rag --limit 6
    .venv/bin/python -m scripts.eval_rag --types multi_hop,comparison
    .venv/bin/python -m scripts.eval_rag --ablation
    .venv/bin/python -m scripts.eval_rag --json eval/results.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openai

import backend.config as config
from backend.context import assemble
from backend.ingestion import _pdfs_in, count_tokens, extract_document
from backend.query_analysis import analyze
from backend.rag import _is_refusal, query
from backend.retrieval import retrieve
from backend.vectorstore import _shared_openai

GOLD_PATH = config.PROJECT_ROOT / "eval" / "gold_set.json"


# ---------------------------------------------------------------------------
# Gold-page resolution
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# Answers spell numbers out ("twenty days"), while labels are written with
# digits ("20"). Scoring the substring literally marked three demonstrably
# correct answers wrong -- a flaw in the metric, not the pipeline. Both forms are
# accepted for the answer-token check.
_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "24": "twenty-four", "28": "twenty-eight",
    "30": "thirty", "42": "forty-two", "60": "sixty", "90": "ninety",
    "180": "one hundred and eighty", "200": "two hundred", "300": "three hundred",
}


def _token_present(token: str, answer_norm: str) -> bool:
    """True if ``token`` appears in the answer, in digit or word form."""
    token_norm = _normalise(token)
    if token_norm in answer_norm:
        return True
    word = _NUMBER_WORDS.get(token_norm)
    if word and word in answer_norm:
        return True
    # And the reverse: a label written as a word, an answer using digits.
    for digits, spelled in _NUMBER_WORDS.items():
        if token_norm == spelled and digits in answer_norm:
            return True
    return False


@dataclass
class FactIndex:
    """Two-resolution index for locating gold facts in the parsed corpus.

    ``pages`` is the precise index. ``spans`` covers adjacent page pairs, for a
    sentence that straddles a page break and therefore exists in full on neither
    page.

    The two are kept SEPARATE and searched precise-first. Merging them
    attributed every fact to both pages of every window containing it, inflating
    the gold set roughly threefold and dragging reported recall from 0.98 to 0.43
    -- a pure measurement artifact that looked exactly like a retrieval
    regression.
    """

    pages: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    spans: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    def page_count(self) -> int:
        return len(self.pages)


def build_fact_index(data_dir: Path) -> FactIndex:
    """Index page texts and adjacent page-pair spans for fact location."""
    index = FactIndex()
    for pdf_path in _pdfs_in(data_dir):
        parsed = extract_document(pdf_path)
        pages = parsed.pages
        for page in pages:
            index.pages.setdefault(_normalise(page.text), []).append(
                (parsed.source, page.page)
            )
        for first, second in zip(pages, pages[1:]):
            key = _normalise(f"{first.text} {second.text}")
            index.spans.setdefault(key, []).extend(
                [(parsed.source, first.page), (parsed.source, second.page)]
            )
    return index


def locate_facts(
    facts: list[str], fact_index: FactIndex
) -> tuple[set[tuple[str, int]], list[str]]:
    """Return the gold ``(source, page)`` set for ``facts`` plus any not found.

    Precise first: if a fact lives wholly on one or more single pages, those
    pages alone are gold. Only a fact found on no single page falls back to the
    page-pair span.
    """
    gold: set[tuple[str, int]] = set()
    missing: list[str] = []
    for fact in facts:
        needle = _normalise(fact)
        exact = {
            location
            for page_text, locations in fact_index.pages.items()
            if needle in page_text
            for location in locations
        }
        if exact:
            gold.update(exact)
            continue
        spanning = {
            location
            for span_text, locations in fact_index.spans.items()
            if needle in span_text
            for location in locations
        }
        if spanning:
            gold.update(spanning)
        else:
            missing.append(fact)
    return gold, missing


# ---------------------------------------------------------------------------
# Answer judging
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You grade a document assistant's answer against a list of \
facts the answer is required to convey. Reply with JSON only.

correct = true only if the answer conveys every required fact, with the right \
values, and states nothing that contradicts them. Wording may differ; numbers \
written as words ("twenty") count as the numeral ("20"). Extra correct detail is \
fine. A hedge or an admission of missing information does not make an otherwise \
complete answer incorrect.

Reply exactly: {"correct": true|false, "missing": ["<required fact>", ...]}"""


def judge_answer(
    question: str, answer: str, required: list[str]
) -> tuple[bool | None, list[str]]:
    """LLM verdict on whether ``answer`` conveys every required fact."""
    if not required:
        return None, []
    try:
        completion = _shared_openai().chat.completions.create(
            model=config.UTILITY_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nRequired facts:\n"
                        + "\n".join(f"- {f}" for f in required)
                        + f"\n\nAnswer:\n{answer}"
                    ),
                },
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except (openai.OpenAIError, json.JSONDecodeError, KeyError, IndexError):
        return None, []
    return bool(payload.get("correct")), [
        str(m) for m in (payload.get("missing") or [])
    ]


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """All metrics for one evaluated question."""

    id: str
    question: str
    expected_type: str
    detected_type: str = ""
    expect_refusal: bool = False
    refused: bool = False

    precision: float | None = None
    recall: float | None = None
    mrr: float | None = None
    source_recall: float | None = None

    context_relevance: float | None = None
    context_completeness: float | None = None

    correct: bool | None = None
    judge_correct: bool | None = None
    fact_coverage: float | None = None
    faithful: bool | None = None
    hallucinated: bool = False
    citation_coverage: float | None = None
    repaired: str = ""

    latency_ms: int = 0
    context_tokens: int = 0
    gold_pages: int = 0
    documents_excluded: int = 0
    missing_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    answer: str = ""


def evaluate_question(entry: dict, fact_index: FactIndex) -> QuestionResult:
    """Run one gold question through the pipeline and score every stage."""
    question = entry["question"]
    required = entry.get("required_facts") or []
    gold_sources = set(entry.get("gold_sources") or [])
    expect_refusal = bool(entry.get("expect_refusal"))

    gold_pages, missing = locate_facts(required, fact_index)

    result = QuestionResult(
        id=entry["id"],
        question=question,
        expected_type=entry.get("type", ""),
        expect_refusal=expect_refusal,
        gold_pages=len(gold_pages),
        missing_labels=missing,
    )
    if missing:
        result.notes.append(f"GOLD DRIFT: {len(missing)} fact string(s) not found")

    # Retrieval and context are measured directly, so a bad answer can be
    # attributed to retrieval rather than generation.
    plan = analyze(question)
    retrieval = retrieve(plan)
    context = assemble(
        retrieval.units,
        max_tokens=min(plan.profile.max_context_tokens, config.MAX_CONTEXT_TOKENS),
        document_order=plan.profile.document_order,
        merge_adjacent=plan.profile.merge_adjacent,
        outline=retrieval.outline,
    )
    result.detected_type = plan.query_type
    result.context_tokens = context.tokens
    result.documents_excluded = len(retrieval.stages.get("documents_excluded") or [])

    if gold_pages:
        retrieved_pairs = [(u.source, u.page) for u in retrieval.units]
        hits = [pair for pair in retrieved_pairs if pair in gold_pages]
        result.precision = len(hits) / len(retrieved_pairs) if retrieved_pairs else 0.0
        result.recall = len(set(hits) & gold_pages) / len(gold_pages)
        rank = next(
            (i for i, pair in enumerate(retrieved_pairs, 1) if pair in gold_pages), None
        )
        result.mrr = 1.0 / rank if rank else 0.0
        retrieved_sources = {u.source for u in retrieval.units}
        result.source_recall = (
            len(gold_sources & retrieved_sources) / len(gold_sources)
            if gold_sources
            else None
        )

        gold_tokens = sum(
            count_tokens(u.text)
            for u in context.units_used
            if (u.source, u.page) in gold_pages
        )
        total_tokens = sum(count_tokens(u.text) for u in context.units_used) or 1
        result.context_relevance = gold_tokens / total_tokens

        context_norm = _normalise(context.text)
        present = sum(1 for f in required if _normalise(f) in context_norm)
        result.context_completeness = present / len(required) if required else None

    response = query(question)
    result.answer = response.answer
    result.refused = _is_refusal(response.answer)
    result.latency_ms = response.diagnostics.get("latency_ms", {}).get("total", 0)
    result.faithful = response.grounding.faithful
    result.repaired = response.grounding.repaired
    result.hallucinated = bool(
        response.grounding.unsupported_claims or response.grounding.unverified_numbers
    )

    if expect_refusal:
        result.correct = result.refused
        if not result.refused:
            result.notes.append("FALSE ANSWER on out-of-scope question")
        return result

    if result.refused:
        result.correct = False
        result.fact_coverage = 0.0
        result.citation_coverage = 0.0
        result.notes.append("FALSE REFUSAL on answerable question")
        return result

    must_include = entry.get("answer_must_include") or []
    missing_tokens: list[str] = []
    if must_include:
        answer_norm = _normalise(result.answer)
        missing_tokens = [
            token for token in must_include if not _token_present(token, answer_norm)
        ]
        result.fact_coverage = (len(must_include) - len(missing_tokens)) / len(
            must_include
        )

    verdict, missing_facts = judge_answer(question, result.answer, required)
    result.judge_correct = verdict

    # Correctness is scored STRUCTURALLY when the label provides the tokens a
    # correct answer must contain, and only falls back to the LLM judge when it
    # does not. The judge proved unreliable on exhaustive questions: asked which
    # of 4 required facts were missing, it returned 7 items -- inventing entries,
    # so any metric built on it was unreproducible.
    if must_include:
        result.correct = not missing_tokens
        if missing_tokens:
            result.notes.append(f"missing from answer: {', '.join(missing_tokens[:6])}")
    else:
        result.correct = verdict
        if missing_facts:
            result.notes.append(f"judge: missing {len(missing_facts)} fact(s)")

    cited = {s.source for s in response.sources}
    result.citation_coverage = (
        len(gold_sources & cited) / len(gold_sources) if gold_sources else None
    )
    if gold_sources and not (gold_sources & cited):
        result.notes.append("NO GOLD SOURCE CITED")

    return result


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _fmt(value: float | None, width: int = 6) -> str:
    return f"{value:>{width}.2f}" if value is not None else " " * (width - 1) + "-"


def summarise(results: list[QuestionResult]) -> dict:
    """Aggregate metrics overall and per query type."""
    # Questions that errored carry no measurements; including them would depress
    # every average as though the pipeline had answered badly.
    errored = [r for r in results if any("EVALUATION ERROR" in n for n in r.notes)]
    scored = [r for r in results if r not in errored]
    answerable = [r for r in scored if not r.expect_refusal]
    out_of_scope = [r for r in scored if r.expect_refusal]

    summary = {
        "questions": len(results),
        "scored": len(scored),
        "errored": len(errored),
        "retrieval_precision": _mean([r.precision for r in answerable]),
        "retrieval_recall": _mean([r.recall for r in answerable]),
        "mrr": _mean([r.mrr for r in answerable]),
        "source_recall": _mean([r.source_recall for r in answerable]),
        "context_relevance": _mean([r.context_relevance for r in answerable]),
        "context_completeness": _mean([r.context_completeness for r in answerable]),
        "answer_correctness": _mean(
            [1.0 if r.correct else 0.0 for r in answerable if r.correct is not None]
        ),
        "fact_coverage": _mean([r.fact_coverage for r in answerable]),
        "faithfulness": _mean(
            [1.0 if r.faithful else 0.0 for r in answerable if r.faithful is not None]
        ),
        "hallucination_rate": _mean([1.0 if r.hallucinated else 0.0 for r in answerable]),
        "citation_coverage": _mean([r.citation_coverage for r in answerable]),
        "repairs_applied": sum(1 for r in answerable if r.repaired),
        "false_refusals": sum(1 for r in answerable if r.refused),
        "correct_refusals": sum(1 for r in out_of_scope if r.refused),
        "out_of_scope_total": len(out_of_scope),
        "type_accuracy": _mean(
            [
                1.0 if r.detected_type == r.expected_type else 0.0
                for r in scored
                if r.expected_type
            ]
        ),
        "median_latency_ms": (
            statistics.median([r.latency_ms for r in scored]) if scored else 0
        ),
        "gold_drift": sum(len(r.missing_labels) for r in scored),
    }

    per_type: dict[str, dict] = {}
    for result in answerable:
        bucket = per_type.setdefault(
            result.expected_type,
            {"n": 0, "precision": [], "recall": [], "completeness": [], "correct": []},
        )
        bucket["n"] += 1
        bucket["precision"].append(result.precision)
        bucket["recall"].append(result.recall)
        bucket["completeness"].append(result.context_completeness)
        if result.correct is not None:
            bucket["correct"].append(1.0 if result.correct else 0.0)

    summary["per_type"] = {
        name: {
            "n": bucket["n"],
            "precision": _mean(bucket["precision"]),
            "recall": _mean(bucket["recall"]),
            "context_completeness": _mean(bucket["completeness"]),
            "correctness": _mean(bucket["correct"]),
        }
        for name, bucket in sorted(per_type.items())
    }
    return summary


def print_report(results: list[QuestionResult], summary: dict) -> None:
    """Print a per-question table and the aggregate summary."""
    print("\n" + "=" * 112)
    print("PER-QUESTION")
    print("=" * 112)
    print(
        f"{'id':26} {'type (detected)':26} {'prec':>6} {'rec':>6} {'ctxrel':>7} "
        f"{'compl':>6} {'corr':>5} {'faith':>6} {'ms':>6}"
    )
    print("-" * 112)
    for r in results:
        type_label = (
            r.detected_type
            if r.detected_type == r.expected_type
            else f"{r.detected_type}<-{r.expected_type}"
        )
        correct = "-" if r.correct is None else ("yes" if r.correct else "NO")
        faithful = "-" if r.faithful is None else ("yes" if r.faithful else "NO")
        print(
            f"{r.id:26} {type_label:26} {_fmt(r.precision)} {_fmt(r.recall)} "
            f"{_fmt(r.context_relevance, 7)} {_fmt(r.context_completeness)} "
            f"{correct:>5} {faithful:>6} {r.latency_ms:>6}"
        )
        for note in r.notes:
            print(f"{'':26} ! {note}")

    print("\n" + "=" * 112)
    print("SUMMARY")
    print("=" * 112)
    for label, key in [
        ("Retrieval precision", "retrieval_precision"),
        ("Retrieval recall", "retrieval_recall"),
        ("MRR", "mrr"),
        ("Source recall", "source_recall"),
        ("Context relevance", "context_relevance"),
        ("Context completeness", "context_completeness"),
        ("Answer correctness", "answer_correctness"),
        ("Answer fact coverage", "fact_coverage"),
        ("Faithfulness", "faithfulness"),
        ("Hallucination rate", "hallucination_rate"),
        ("Citation coverage", "citation_coverage"),
        ("Query-type accuracy", "type_accuracy"),
    ]:
        print(f"  {label:24} {_fmt(summary.get(key))}")
    print(f"  {'Grounding repairs':24} {summary['repairs_applied']:>6}")
    print(f"  {'False refusals':24} {summary['false_refusals']:>6}")
    print(
        f"  {'Correct refusals':24} "
        f"{summary['correct_refusals']:>3}/{summary['out_of_scope_total']}"
    )
    print(f"  {'Median latency (ms)':24} {summary['median_latency_ms']:>6.0f}")
    if summary["errored"]:
        print(f"  {'ERRORED (excluded)':24} {summary['errored']:>6}")
    if summary["gold_drift"]:
        print(f"  {'GOLD DRIFT (facts lost)':24} {summary['gold_drift']:>6}")

    print("\n  Per query type:")
    print(f"    {'type':16} {'n':>3} {'prec':>6} {'rec':>6} {'compl':>6} {'corr':>6}")
    for name, bucket in summary["per_type"].items():
        print(
            f"    {name:16} {bucket['n']:>3} {_fmt(bucket['precision'])} "
            f"{_fmt(bucket['recall'])} {_fmt(bucket['context_completeness'])} "
            f"{_fmt(bucket['correctness'])}"
        )


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

ABLATIONS = [
    ("full pipeline", {}),
    ("no reranking", {"ENABLE_RERANK": False}),
    ("no hybrid (dense only)", {"ENABLE_HYBRID": False}),
    ("no decomposition", {"ENABLE_DECOMPOSITION": False}),
    ("no parent expansion", {"ENABLE_PARENT_EXPANSION": False}),
    ("no document routing", {"ENABLE_DOC_ROUTING": False}),
    ("no grounding repair", {"ENABLE_GROUNDING_REPAIR": False}),
]


def run_ablation(entries: list[dict], fact_index: FactIndex) -> None:
    """Measure each stage's contribution by disabling it.

    Reports rather than asserts: a stage that does not move the numbers on this
    corpus should be called out, not defended.
    """
    print("\n" + "=" * 112)
    print("ABLATION")
    print("=" * 112)
    print(
        f"{'configuration':26} {'prec':>6} {'rec':>6} {'compl':>6} {'corr':>6} "
        f"{'faith':>6} {'ms':>7}"
    )
    print("-" * 112)

    original = {key: getattr(config, key) for _, flags in ABLATIONS for key in flags}
    try:
        for label, flags in ABLATIONS:
            for key, value in original.items():
                setattr(config, key, value)
            for key, value in flags.items():
                setattr(config, key, value)

            results = [evaluate_question(entry, fact_index) for entry in entries]
            summary = summarise(results)
            print(
                f"{label:26} {_fmt(summary['retrieval_precision'])} "
                f"{_fmt(summary['retrieval_recall'])} "
                f"{_fmt(summary['context_completeness'])} "
                f"{_fmt(summary['answer_correctness'])} "
                f"{_fmt(summary['faithfulness'])} "
                f"{summary['median_latency_ms']:>7.0f}"
            )
    finally:
        for key, value in original.items():
            setattr(config, key, value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.PROJECT_ROOT / "data")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N.")
    parser.add_argument("--types", default=None, help="Comma-separated query types.")
    parser.add_argument("--ablation", action="store_true", help="Run stage ablation.")
    parser.add_argument("--json", type=Path, default=None, help="Write results JSON.")
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    entries = gold["questions"]

    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        entries = [e for e in entries if e.get("type") in wanted]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("No questions selected.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Corpus     : {args.data_dir}")
    print(f"Questions  : {len(entries)}")
    print(f"Embed model: {config.EMBED_MODEL} | chat: {config.CHAT_MODEL}")
    print(
        f"Pipeline   : hybrid={config.ENABLE_HYBRID} rerank={config.ENABLE_RERANK} "
        f"decompose={config.ENABLE_DECOMPOSITION} "
        f"routing={config.ENABLE_DOC_ROUTING} repair={config.ENABLE_GROUNDING_REPAIR}"
    )
    print("\nIndexing corpus fact locations for gold-page resolution ...")
    fact_index = build_fact_index(args.data_dir)
    print(f"  {fact_index.page_count()} pages indexed.")

    if args.ablation:
        run_ablation(entries, fact_index)
        return

    started = time.time()
    results: list[QuestionResult] = []
    for i, entry in enumerate(entries, start=1):
        print(f"  [{i}/{len(entries)}] {entry['id']} ...", flush=True)
        try:
            results.append(evaluate_question(entry, fact_index))
        except Exception as exc:  # noqa: BLE001
            # One transient failure must not discard a whole run. A 30-question
            # evaluation is ~20 minutes of API calls; aborting on a single
            # dropped connection loses every completed measurement.
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

    summary = summarise(results)
    print_report(results, summary)
    print(f"\nCompleted in {time.time() - started:.0f}s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(r) for r in results]}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
