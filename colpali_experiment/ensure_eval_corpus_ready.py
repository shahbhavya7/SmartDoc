"""Precondition check for a ColPali eval pass (colpali branch experiment).

Before running eval/gold_set.json against RETRIEVAL_BACKEND=colpali, every
document the gold set references must have ColPali status 'ready' for the
user the eval harness logs in as (dev-user-0001 by default -- see
eval/eval_tool/config.py's API_EMAIL/API_PASSWORD). A "still indexing"
response scored as a wrong answer would penalize ColPali for a timing issue,
not a genuine retrieval/generation gap, and would corrupt the comparison --
so this script FAILS LOUDLY and lists exactly which documents aren't ready,
rather than letting an eval run start against a half-indexed corpus.

Usage:
    # Report-only: what's ready, pending, failed, or never even started.
    .venv/bin/python -m colpali_experiment.ensure_eval_corpus_ready --check-only

    # Trigger ingestion for anything missing, then poll until every gold-set
    # document is ready (or genuinely failed) -- CPU-only, low priority, same
    # resource-conscious defaults as the rest of this experiment.
    .venv/bin/python -m colpali_experiment.ensure_eval_corpus_ready \\
        --user dev-user-0001 --poll-interval 15 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from backend import db as backend_db
from colpali_experiment import ingest, store


def _gold_set_documents(gold_set_path: str) -> set[str]:
    """Filenames referenced by the gold set, excluding null-source (OOS/edge/
    new-document/consistency) questions -- those have no corpus document to
    check readiness for.
    """
    entries = json.loads(open(gold_set_path).read())
    return {e["expected_source"] for e in entries if e.get("expected_source")}


def _status_by_filename(user_id: str) -> dict[str, str]:
    return {
        s["filename"]: s["status"]
        for s in store.get_ingest_statuses_for_user(user_id)
    }


def check(user_id: str, gold_set_path: str) -> dict[str, list[str]]:
    """Returns {"ready": [...], "pending": [...], "failed": [...], "missing": [...]}.

    'missing' means no status row exists at all -- this document has never
    been through the /upload ColPali fan-out (Phase 3) or a manual ingest, so
    there is nothing to poll; it needs to be triggered first.
    """
    wanted = _gold_set_documents(gold_set_path)
    owned = {r["filename"]: r["id"] for r in backend_db.list_documents(user_id)}
    statuses = _status_by_filename(user_id)

    result = {"ready": [], "pending": [], "failed": [], "missing": [], "not_owned": []}
    for filename in sorted(wanted):
        if filename not in owned:
            result["not_owned"].append(filename)
            continue
        status = statuses.get(filename)
        if status is None:
            result["missing"].append(filename)
        elif status == "ready":
            result["ready"].append(filename)
        elif status == "pending":
            result["pending"].append(filename)
        else:
            result["failed"].append(filename)
    return result


def trigger_missing(user_id: str, filenames: list[str]) -> None:
    """Synchronously ingest each filename not yet started.

    Not via /upload's background fan-out here -- these documents already
    exist in the hybrid pipeline (uploaded long ago), so there is no upload
    happening to piggyback on. Calling colpali_experiment.ingest directly is
    the correct, isolated equivalent: it only reads the existing `documents`
    row and writes to this experiment's own store, exactly like a fan-out
    task would, just triggered explicitly instead of from /upload.
    """
    owned = {r["filename"]: r["id"] for r in backend_db.list_documents(user_id)}
    for filename in filenames:
        document_id = owned[filename]
        store.set_ingest_status(document_id, user_id, filename, status="pending")
        try:
            ingest.ingest_document_for_upload(document_id, user_id, filename)
        except Exception as exc:  # noqa: BLE001 - reported via status, not raised
            print(f"  ERROR ingesting {filename}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user", default="dev-user-0001")
    p.add_argument("--gold-set", default="eval/gold_set.json")
    p.add_argument(
        "--check-only", action="store_true",
        help="Report status and exit -- never triggers ingestion.",
    )
    p.add_argument(
        "--poll-interval", type=int, default=15,
        help="Seconds between status checks while waiting for pending documents.",
    )
    p.add_argument(
        "--timeout", type=int, default=3600,
        help="Give up waiting after this many seconds and fail loudly.",
    )
    args = p.parse_args(argv)

    result = check(args.user, args.gold_set)
    total = sum(len(v) for v in result.values())
    print(f"Gold-set documents for user {args.user!r}: {total} referenced.")
    for bucket, filenames in result.items():
        if filenames:
            print(f"  {bucket}: {len(filenames)}")
            for f in filenames:
                print(f"    - {f}")

    if result["not_owned"]:
        print(
            "\nFAIL: some gold-set documents are not owned by this user at all "
            "in the hybrid pipeline -- nothing to ColPali-ingest until they "
            "exist there.",
            file=sys.stderr,
        )
        return 1

    if args.check_only:
        not_ready = result["pending"] + result["failed"] + result["missing"]
        if not_ready:
            print(
                f"\nNOT READY: {len(not_ready)} document(s) are not "
                "colpali_indexed='ready'. Re-run without --check-only to "
                "trigger/wait, or investigate 'failed' entries directly.",
                file=sys.stderr,
            )
            return 1
        print("\nREADY: every gold-set document is colpali_indexed='ready'.")
        return 0

    if result["missing"]:
        print(f"\nTriggering ColPali ingestion for {len(result['missing'])} document(s)...")
        trigger_missing(args.user, result["missing"])

    if result["failed"]:
        print(
            f"\nFAIL: {len(result['failed'])} document(s) have status='failed' "
            "from a prior attempt -- investigate before proceeding (see "
            "colpali_ingest_status.error in colpali_store.db). Not retrying "
            "automatically: a repeated failure is a real bug to fix, not a "
            "timing issue to wait out.",
            file=sys.stderr,
        )
        for f in result["failed"]:
            print(f"    - {f}", file=sys.stderr)
        return 1

    deadline = time.monotonic() + args.timeout
    while True:
        result = check(args.user, args.gold_set)
        still_pending = result["pending"]
        if not still_pending and not result["missing"]:
            if result["failed"]:
                print(
                    f"\nFAIL: {len(result['failed'])} document(s) failed during "
                    "this wait.",
                    file=sys.stderr,
                )
                for f in result["failed"]:
                    print(f"    - {f}", file=sys.stderr)
                return 1
            print(f"\nREADY: all {len(result['ready'])} gold-set document(s) are ready.")
            return 0
        if time.monotonic() > deadline:
            print(
                f"\nFAIL: timed out after {args.timeout}s waiting for "
                f"{len(still_pending)} document(s) still pending:",
                file=sys.stderr,
            )
            for f in still_pending:
                print(f"    - {f}", file=sys.stderr)
            return 1
        print(f"  still pending: {still_pending} -- waiting {args.poll_interval}s...")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
