"""Acceptance check: Large_Multi_Page_Tables_Test.pdf's one continuous
60-employee-row table, spanning all 6 of its pages, should cluster into a
single visual table group (or a small number of groups whose boundaries make
sense) -- verified by printing the per-page cluster assignment.

Note on the document's real shape (checked against the actual file, not
assumed): it is 6 pages, not ~170. The row numbers MP0033/MP0034,
MP0067/MP0068, MP0101/MP0102, MP0135/MP0136, MP0169/MP0170 are ROW-id
boundaries that fall exactly at each of its 5 internal page breaks (1|2, 2|3,
3|4, 4|5, 5|6) -- i.e. every page break in this document IS one of those
listed boundaries, which is what confirms this is the intended fixture.

Usage:
    .venv/bin/python -m colpali_experiment.verify_table_clustering \\
        --user 83175468-5c27-4338-8be6-13470567047f \\
        --document-id 9ef261a6-db3d-46d7-914c-64d02fdb962b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend import db as backend_db
from backend.documents import DATA_DIR, stored_path
from colpali_experiment.table_clustering import cluster_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument(
        "--vision-confirm", action="store_true",
        help="Confirm ambiguous pairs with one vision-LLM call each.",
    )
    args = parser.parse_args()

    record = backend_db.get_document(args.user, args.document_id)
    if record is None:
        print(f"No document {args.document_id!r} for user {args.user!r}.")
        return 1

    pdf_path = None
    for candidate in (
        stored_path(args.user, record["filename"]),
        DATA_DIR / Path(record["filename"]).name,
    ):
        if candidate.is_file():
            pdf_path = candidate
            break

    result = cluster_document(
        args.document_id, pdf_path=pdf_path, use_vision_confirmation=args.vision_confirm
    )

    print(f"Document: {record['filename']} ({args.document_id})\n")
    print("Page-to-page continuity scores:")
    for pair in result.pair_scores:
        flag = "CONTINUES" if pair.is_continuation else "breaks"
        print(
            f"  page {pair.page_a:>3} -> {pair.page_b:<3} "
            f"normalized={pair.normalized_score:.4f}  [{flag}]  ({pair.confirmed_by})"
        )

    print("\nPer-page cluster assignment:")
    for page in sorted(result.groups):
        print(f"  page {page:>3}: group {result.groups[page]}")

    distinct_groups = sorted(set(result.groups.values()))
    breaks = [p.page_a for p in result.pair_scores if not p.is_continuation]
    print(f"\n{len(distinct_groups)} distinct visual table group(s) across "
          f"{len(result.groups)} pages (breaks after page(s): {breaks or 'none'}).")
    print(
        "This script only reports what was detected -- it does not assert "
        "'should be 1 group'. Compare against this document's KNOWN structure "
        "(e.g. Large_Multi_Page_Tables_Test.pdf's single table spanning all "
        "pages vs. a prose document with no continued table) to judge PASS/FAIL."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
