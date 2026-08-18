"""CLI trigger for ColPali ingestion. Separate from /upload, on purpose (the
brief's hard constraint): this only ever reads existing document rows.

Usage:
    .venv/bin/python -m colpali_experiment.run_ingest --user dev-user-0001
    .venv/bin/python -m colpali_experiment.run_ingest --user dev-user-0001 --document-id <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from colpali_experiment import ingest, store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Existing user_id to ingest for.")
    parser.add_argument(
        "--document-id", default=None, help="Ingest only this document (default: all)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-embed even if already stored."
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Embed only the first N pages (proof-of-concept / resource-light runs).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Pages embedded per forward pass. Default 1 -- lightest possible load.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.document_id:
        results = [
            ingest.ingest_document(
                args.document_id, args.user,
                batch_size=args.batch_size, max_pages=args.max_pages,
            )
        ]
    else:
        results = ingest.ingest_all_for_user(args.user, skip_existing=not args.force)

    print(json.dumps(results, indent=2))
    print(json.dumps({"totals": store.stats()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
