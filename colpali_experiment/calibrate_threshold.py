"""Print the real page-to-page MaxSim score distribution, so the continuity
threshold in config.py is calibrated on data rather than guessed.

Usage:
    .venv/bin/python -m colpali_experiment.calibrate_threshold --user dev-user-0001
"""

from __future__ import annotations

import argparse
import sys

from backend import db as backend_db
from colpali_experiment.table_clustering import pairwise_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    args = parser.parse_args()

    all_scores = []
    for record in backend_db.list_documents(args.user):
        pairs = pairwise_scores(record["id"])
        for pair in pairs:
            all_scores.append((record["filename"], pair))

    if not all_scores:
        print("No stored page embeddings found -- run colpali_experiment.run_ingest first.")
        return 1

    all_scores.sort(key=lambda t: t[1].normalized_score)
    print(f"{'filename':<45} {'pages':<8} {'normalized':>10} {'raw':>10}")
    for filename, pair in all_scores:
        print(
            f"{filename:<45} {pair.page_a:>3}->{pair.page_b:<3} "
            f"{pair.normalized_score:>10.4f} {pair.raw_score:>10.4f}"
        )

    values = [p.normalized_score for _, p in all_scores]
    print(f"\nmin={min(values):.4f}  max={max(values):.4f}  "
          f"n={len(values)}  median={sorted(values)[len(values)//2]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
