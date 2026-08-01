"""Ingest entrypoint: parse data/ PDFs and index them in Chroma.

Per document, this parses structure (headings, tables, page ranges), builds
parent and child chunks, embeds the children, and replaces any previously
indexed version of that document.

Replacement, not just idempotency
---------------------------------
Re-running over an unchanged corpus is free: each document's content hash is
compared against what is indexed, and unchanged documents are skipped without
embedding calls. When a document HAS changed, every chunk of the old version is
deleted before the new ones are written -- so an edited or shortened policy
cannot leave superseded text behind in the index, which upsert-by-id alone did.

Usage:
    .venv/bin/python -m backend.ingest [--reset] [--force] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import backend.config as config
from backend.ingestion import _pdfs_in, build_chunks, extract_document
from backend.vectorstore import (
    collection_stats,
    get_collection,
    indexed_hashes,
    ingest_documents,
    reset_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.PROJECT_ROOT / "data",
        help="Directory of PDFs to ingest (default: data/).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the whole collection before ingesting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed documents even if their content hash is unchanged.",
    )
    args = parser.parse_args()

    print(f"Data directory : {args.data_dir}")
    print(f"Collection     : {config.CHROMA_COLLECTION} at {config.CHROMA_DIR}")
    print(f"Embedding model: {config.EMBED_MODEL}")
    print(
        f"Chunking       : child {config.CHILD_CHUNK_SIZE}/"
        f"{config.CHILD_CHUNK_OVERLAP} tokens, parent {config.PARENT_CHUNK_SIZE}\n"
    )

    if args.reset:
        print("Resetting collection ...")
        reset_collection()

    collection = get_collection()
    known = {} if args.reset else indexed_hashes(collection)

    total_children = 0
    total_parents = 0
    skipped = 0
    start = time.time()

    for pdf_path in _pdfs_in(args.data_dir):
        parsed = extract_document(pdf_path)

        if not args.force and known.get(parsed.source) == parsed.content_hash:
            print(f"  {parsed.source:48} unchanged, skipped")
            skipped += 1
            continue

        parents, children = build_chunks(parsed)
        if not children:
            # A PDF with no extractable text layer (e.g. a pure scan) yields
            # nothing; say so rather than reporting a silent success.
            print(f"  {parsed.source:48} NO TEXT EXTRACTED - skipped (scanned?)")
            continue

        indexed = ingest_documents(children, parents=parents)
        total_children += indexed
        total_parents += len(parents)
        print(
            f"  {parsed.source:48} {parsed.page_count:3}p  "
            f"{len(parents):3} parents  {indexed:4} chunks"
        )

    elapsed = time.time() - start
    print(
        f"\nIndexed {total_children} chunks ({total_parents} parents) in "
        f"{elapsed:.1f}s; {skipped} document(s) unchanged."
    )
    print(f"Collection stats: {collection_stats()}")


if __name__ == "__main__":
    main()
