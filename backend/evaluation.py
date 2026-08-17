"""Service layer for the evaluation harness: reading past runs, starting new ones.

This module exists so ``main.py`` stays a thin HTTP layer. It owns three things:

  * **Reading results.** Runs are JSON files under ``eval/results/``; this
    module lists and loads them, and derives the "how it was scored" explanation
    that the UI renders, so the frontend never re-implements scoring rules.
  * **Per-user ownership.** A run is tagged with the user who started it and is
    only ever listed back to that user. Evaluation asks questions against the
    caller's own documents, so showing one user another's numbers would be
    showing them results for a corpus they cannot see.
  * **Running an eval in the background.** A full run is minutes long, so the
    HTTP call returns a run id immediately and progress is polled.

The runner itself is NOT reimplemented here. It is imported from
``eval.eval_tool``, the same code path the CLI uses, so a run started from the
browser and a run started from a terminal execute identical logic. The one thing
this module does differently is talk to the API over loopback with the caller's
own token, which keeps the harness's "read-only, over HTTP" property intact.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import pathlib
import threading
import uuid
from typing import Any

import backend.config as config

logger = logging.getLogger(__name__)

EVAL_DIR = config.PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
GOLD_SET_PATH = EVAL_DIR / "gold_set.json"
CALIBRATION_PATH = RESULTS_DIR / "calibration.json"
# Uploaded test sets, kept per user so one user's upload is never run or listed
# for another.
UPLOADS_DIR = EVAL_DIR / "uploads"


class EvalError(Exception):
    """A request against the evaluation harness could not be served."""


# --------------------------------------------------------------------------
# Reading results
# --------------------------------------------------------------------------


def _run_files() -> list[pathlib.Path]:
    if not RESULTS_DIR.exists():
        return []
    # calibration.json is not a run.
    return sorted(
        (p for p in RESULTS_DIR.glob("*.json") if p.name != "calibration.json"),
        key=lambda p: p.name,
        reverse=True,
    )


def _load(path: pathlib.Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Unreadable eval result file: %s", path)
        return None
    if not isinstance(payload, dict) or "summary" not in payload:
        return None
    return payload


def _owner_of(payload: dict) -> str | None:
    return (payload.get("meta") or {}).get("user_id")


def list_runs(user_id: str, limit: int = 25) -> list[dict]:
    """Summaries of this user's past runs, newest first.

    Runs recorded before per-user tagging existed (the CLI runs that produced
    the shipped baseline) carry no ``user_id``. Those are surfaced to everyone
    and flagged ``shared: true`` -- they are the project's own reference
    numbers, and hiding them would leave a new account staring at an empty page
    with no way to see what the system scores.
    """
    out: list[dict] = []
    for path in _run_files():
        payload = _load(path)
        if payload is None:
            continue
        owner = _owner_of(payload)
        if owner is not None and owner != user_id:
            continue
        meta = payload.get("meta") or {}
        summary = payload.get("summary") or {}
        out.append(
            {
                "run_id": path.stem,
                "timestamp": meta.get("timestamp") or path.stem,
                "shared": owner is None,
                "label": meta.get("label", ""),
                "gold_set": meta.get("gold_set", ""),
                "question_count": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "pass_rate": summary.get("pass_rate", 0.0),
                "mean_similarity": summary.get("mean_similarity"),
                "threshold": meta.get("threshold"),
            }
        )
        if len(out) >= limit:
            break
    return out


def get_run(user_id: str, run_id: str) -> dict:
    """One run in full, including every per-question result."""
    # Defend the path join: run_id reaches this from the URL.
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise EvalError("Invalid run id.")
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise EvalError(f"No such evaluation run: {run_id}")
    payload = _load(path)
    if payload is None:
        raise EvalError(f"Evaluation run {run_id} is unreadable.")
    owner = _owner_of(payload)
    if owner is not None and owner != user_id:
        raise EvalError(f"No such evaluation run: {run_id}")
    return payload


def latest_run(user_id: str) -> dict | None:
    runs = list_runs(user_id, limit=1)
    return get_run(user_id, runs[0]["run_id"]) if runs else None


def calibration() -> dict | None:
    """The threshold calibration, if one has been run.

    ``current_threshold`` is overwritten with the LIVE configured value rather
    than served as recorded. The file stores whatever the threshold happened to
    be when calibration ran -- which is, by construction, the value calibration
    was about to replace. Serving that stale number would caption the panel with
    a threshold no run actually uses, and this panel exists precisely to explain
    where the real one came from. The measured distributions are untouched.
    """
    if not CALIBRATION_PATH.exists():
        return None
    try:
        data = json.loads(CALIBRATION_PATH.read_text())
    except (OSError, ValueError):
        return None

    from eval.eval_tool import config as eval_config

    data["current_threshold"] = eval_config.THRESHOLD
    data["calibrated_threshold"] = data.get("proposed_threshold")
    return data


def gold_set_overview() -> dict:
    """Category counts for the shipped gold set, for the coverage display."""
    from eval.eval_tool import schema

    try:
        gold = schema.load_gold_set(GOLD_SET_PATH)
    except Exception as exc:  # noqa: BLE001 - surfaced as a message, not a 500
        return {"error": str(exc), "categories": [], "total": 0}

    counts = schema.category_counts(gold)
    return {
        "total": len(gold),
        "minimum_per_category": schema.MIN_PER_CATEGORY,
        "categories": [
            {
                "name": name,
                "count": count,
                "meets_minimum": count >= schema.MIN_PER_CATEGORY,
                "scored_by": _scoring_mode(name),
            }
            for name, count in counts.items()
        ],
    }


def _scoring_mode(category: str) -> str:
    """Which rule decides pass/fail for a category. Mirrors runner._score."""
    from eval.eval_tool import schema

    if category in schema.EDGE_CATEGORIES:
        return "behaviour"
    if category in schema.OUT_OF_SCOPE_CATEGORIES:
        return "refusal"
    if category == "out_of_scope_partial":
        return "refusal+exact"
    if category == "consistency_pair":
        return "self_similarity"
    if category in schema.COMPLETENESS_CATEGORIES:
        return "similarity+completeness"
    if category in schema.TABLE_FORMAT_CATEGORIES:
        return "similarity+table"
    return "similarity+exact"


# --------------------------------------------------------------------------
# Uploaded test sets
# --------------------------------------------------------------------------


def save_test_set(user_id: str, filename: str, content: bytes) -> dict:
    """Validate an uploaded gold set and store it against this user.

    Validation happens BEFORE the file is accepted, so a malformed set fails
    immediately with a message naming the problem rather than at run time, half
    way through a job the user is watching a spinner for.
    """
    from eval.eval_tool import schema

    safe = pathlib.Path(filename or "test_set.json").name
    if not safe.lower().endswith((".json", ".csv")):
        raise EvalError("Test set must be a .json or .csv file.")

    user_dir = UPLOADS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = user_dir / f"{stamp}-{safe}"
    target.write_bytes(content)

    try:
        gold = schema.load_gold_set(target)
    except schema.GoldSetError as exc:
        target.unlink(missing_ok=True)
        raise EvalError(str(exc)) from exc
    if not gold:
        target.unlink(missing_ok=True)
        raise EvalError("That test set contains no questions.")

    from collections import Counter

    counts = Counter(e["category"] for e in gold)
    return {
        "test_set_id": target.name,
        "filename": safe,
        "question_count": len(gold),
        "categories": [
            {"name": name, "count": count} for name, count in sorted(counts.items())
        ],
    }


def _resolve_test_set(user_id: str, test_set_id: str | None) -> pathlib.Path:
    if not test_set_id:
        return GOLD_SET_PATH
    safe = pathlib.Path(test_set_id).name
    path = UPLOADS_DIR / user_id / safe
    if not path.exists():
        raise EvalError(f"No such uploaded test set: {safe}")
    return path


# --------------------------------------------------------------------------
# Running an evaluation
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RunState:
    """Progress for one in-flight evaluation."""

    job_id: str
    user_id: str
    status: str = "queued"  # queued | running | done | error
    phase: str = ""
    total: int = 0
    completed: int = 0
    run_id: str | None = None
    error: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_JOBS: dict[str, RunState] = {}
_JOBS_LOCK = threading.Lock()
# One evaluation at a time per process: a run drives the same API this server is
# serving, so two concurrent runs would compete for the same workers and measure
# each other's queueing rather than the pipeline.
_RUN_LOCK = threading.Lock()


def get_job(user_id: str, job_id: str) -> RunState:
    with _JOBS_LOCK:
        state = _JOBS.get(job_id)
    if state is None or state.user_id != user_id:
        raise EvalError(f"No such evaluation job: {job_id}")
    return state


def active_job(user_id: str) -> RunState | None:
    with _JOBS_LOCK:
        for state in _JOBS.values():
            if state.user_id == user_id and state.status in ("queued", "running"):
                return state
    return None


def start_run(
    user_id: str,
    token: str,
    base_url: str,
    *,
    test_set_id: str | None = None,
    categories: list[str] | None = None,
    skip_consistency_wait: bool = True,
    label: str = "",
) -> RunState:
    """Kick off an evaluation in a background thread; return its initial state.

    ``token`` is the caller's own bearer token, handed to the harness so it
    authenticates as this user over loopback. That is what keeps the run scoped
    to the caller's documents without the harness needing any notion of identity
    of its own.

    ``base_url`` is derived from the incoming request rather than from config,
    because the port this server is actually listening on is a launch-time
    choice (``API_PORT`` in run.sh, or a uvicorn flag) that no module-level
    constant reliably knows.
    """
    if active_job(user_id) is not None:
        raise EvalError(
            "An evaluation is already running for this account. Wait for it to "
            "finish before starting another."
        )

    gold_path = _resolve_test_set(user_id, test_set_id)
    job = RunState(
        job_id=uuid.uuid4().hex,
        user_id=user_id,
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job

    thread = threading.Thread(
        target=_execute,
        args=(job, token, base_url, gold_path, categories, skip_consistency_wait, label),
        name=f"eval-{job.job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return job


def _execute(
    job: RunState,
    token: str,
    base_url: str,
    gold_path: pathlib.Path,
    categories: list[str] | None,
    skip_consistency_wait: bool,
    label: str,
) -> None:
    """Run the harness. Exceptions become job.error, never a dead thread."""
    if not _RUN_LOCK.acquire(blocking=False):
        job.status = "error"
        job.error = "Another evaluation is already running on this server."
        job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        return

    try:
        # Imported lazily: the eval package pulls in the OpenAI client and the
        # gold set, none of which should be a cost of merely starting the API.
        from eval.eval_tool import config as eval_config
        from eval.eval_tool import report, schema
        from eval.eval_tool.api_client import SmartDocClient
        from eval.eval_tool.runner import EvalRunner

        job.status = "running"
        job.phase = "loading test set"

        gold = schema.load_gold_set(gold_path)
        if categories:
            gold = [e for e in gold if e["category"] in set(categories)]
        if not gold:
            raise EvalError("No questions to run after filtering.")
        job.total = len(gold)

        # Talk to ourselves over loopback with the caller's token. Using the
        # real HTTP path (rather than calling query() directly) is deliberate:
        # it is what makes these numbers describe what a user experiences.
        client = SmartDocClient(base_url=base_url)
        client._token = token  # noqa: SLF001 - deliberate: reuse the caller's auth
        client._session.headers["Authorization"] = f"Bearer {token}"

        def progress(message: str) -> None:
            job.phase = message.strip()
            # runner logs "    asked 40/115"; surface the count for a real bar.
            if "asked" in message and "/" in message:
                try:
                    fraction = message.split("asked", 1)[1].strip()
                    done, total = fraction.split("/")
                    job.completed = int(done.strip())
                    job.total = max(job.total, int(total.strip()))
                except (ValueError, IndexError):
                    pass

        runner = EvalRunner(
            client,
            gold,
            consistency_wait=0 if skip_consistency_wait else None,
            progress=progress,
        )
        results = runner.run()

        job.phase = "writing report"
        json_path, _text_path, _text = report.save(
            results,
            threshold=runner.threshold,
            consistency_threshold=runner.consistency_threshold,
            meta={
                "gold_set": str(gold_path),
                "question_count": len(gold),
                "consistency_wait_seconds": runner.consistency_wait,
                # Ownership, so this run is listed back only to its author.
                "user_id": job.user_id,
                "label": label,
                "source": "web",
            },
        )
        job.run_id = json_path.stem
        job.completed = len(results)
        job.status = "done"
        job.phase = "complete"
    except Exception as exc:  # noqa: BLE001 - reported to the poller
        logger.exception("Evaluation job %s failed", job.job_id)
        job.status = "error"
        job.error = str(exc)
    finally:
        job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        _RUN_LOCK.release()


# --------------------------------------------------------------------------
# The plain-English explanation the UI renders
# --------------------------------------------------------------------------

# Kept server-side, next to the code it describes, so the explanation cannot
# drift away from the actual scoring rules the way a hardcoded copy in the
# frontend would.
METHOD_EXPLAINER: dict[str, Any] = {
    "summary": (
        "Every question is sent to the real running system, exactly as a user "
        "would ask it. The answer that comes back is then compared against a "
        "known-correct answer written by hand from the source documents."
    ),
    "steps": [
        {
            "title": "Ask the real system",
            "body": (
                "The question goes to POST /ask over HTTP -- the same endpoint "
                "the chat page uses. Nothing is mocked or shortcut, so the score "
                "reflects what you would actually get."
            ),
        },
        {
            "title": "Turn both answers into numbers",
            "body": (
                "The system's answer and the known-correct answer are each "
                "converted into a list of numbers (an 'embedding') that captures "
                "their meaning, using the same model the search pipeline uses."
            ),
        },
        {
            "title": "Measure how close they are",
            "body": (
                "Cosine similarity scores the two from 0 to 1. Around 1 means "
                "'these say the same thing'; near 0 means 'these are unrelated'. "
                "Wording may differ -- only meaning is compared."
            ),
        },
        {
            "title": "Check the actual facts separately",
            "body": (
                "Similarity alone would accept a wrong number in a right-sounding "
                "sentence, so any answer containing a number, code, or ID must "
                "ALSO contain the exact expected value. This check can fail an "
                "answer on its own, no matter how high its similarity."
            ),
        },
        {
            "title": "Decide pass or fail",
            "body": (
                "An answer passes when it is similar enough AND its exact values "
                "match. Some categories add a rule: lists must be complete, "
                "comparisons must render a table, and out-of-scope questions must "
                "be declined rather than answered."
            ),
        },
    ],
    "why_exact_match": (
        "'The team scored 78' and 'The team scored 87' are almost identical to a "
        "similarity score -- the sentences differ by one digit. Measured on this "
        "corpus, deliberately wrong numbers still scored 0.81 on average, versus "
        "0.86 for correct answers. No threshold can separate those. That is why "
        "the exact-value check is mandatory rather than a bonus."
    ),
    "metrics": [
        {
            "name": "Similarity",
            "plain": "How close in meaning the answer is to the correct one, from 0 to 1.",
            "detail": "Cosine similarity between text-embedding-3-small embeddings.",
        },
        {
            "name": "Exact match",
            "plain": "Whether the specific number, code, or ID actually appears in the answer.",
            "detail": (
                "Values are matched as whole tokens, so '5' is not credited by "
                "'15' and 'E-05' is not credited by 'E-06'. Written-out numerals "
                "count too: 'twenty' satisfies '20'."
            ),
        },
        {
            "name": "Completeness",
            "plain": "For list questions, how many of the expected items were actually mentioned.",
            "detail": "Reported as partial credit (e.g. 5/7) alongside pass/fail.",
        },
        {
            "name": "Consistency",
            "plain": "Whether asking the same question twice gives the same answer.",
            "detail": (
                "The two answers are compared against EACH OTHER rather than "
                "against the expected answer, so this measures stability, not "
                "correctness."
            ),
        },
    ],
    "categories_note": (
        "Not every question is scored the same way. Out-of-scope questions pass "
        "by being correctly declined, and edge cases (blank input, gibberish, "
        "prompt injection) pass by behaving sensibly -- similarity is meaningless "
        "for both, so it is recorded but never decides the outcome."
    ),
}
