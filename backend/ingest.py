"""Idempotent ingest entrypoint: chunk data/ PDFs and write them to Chroma.

Default behavior upserts by deterministic chunk id (``source:chunk_index``),
so re-running this script does not duplicate the collection. Pass
``--reset`` to instead delete and rebuild the collection from scratch.

Usage:
    .venv/bin/python -m backend.ingest [--reset] [--data-dir DIR]

This uses the REAL OpenAI embedding path (``backend.vectorstore.openai_embed_fn``)
and will raise a clear error if OPENAI_API_KEY is not set in .env -- it never
silently substitutes a fake embedder.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.config import CHROMA_COLLECTION, CHROMA_DIR, PROJECT_ROOT
from backend.ingestion import load_and_chunk_directory
from backend.vectorstore import collection_stats, ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory of PDFs to ingest (default: data/).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the collection instead of upserting.",
    )
    args = parser.parse_args()

    print(f"Loading + chunking PDFs from: {args.data_dir}")
    chunks = load_and_chunk_directory(args.data_dir)
    print(f"Chunked into {len(chunks)} documents.")

    print(
        f"{'Resetting' if args.reset else 'Upserting into'} collection "
        f"'{CHROMA_COLLECTION}' at {CHROMA_DIR} ..."
    )
    start = time.time()
    upserted = ingest_documents(chunks, reset=args.reset)
    elapsed = time.time() - start

    stats = collection_stats()
    print(f"Upserted {upserted} chunks in {elapsed:.1f}s.")
    print(f"Collection stats: {stats}")


if __name__ == "__main__":
    main()
