"""Eyeball script: run ingestion over data/ and print a sample of chunks.

Usage:
    .venv/bin/python -m backend.print_chunks [data_dir]

Prints the total chunk count and the first ~5 chunks with their metadata
and a text preview, so ingestion output can be visually verified. Also
prints token-length stats (min/median/max) to show CHUNK_SIZE is honored.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import median

import tiktoken

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, PROJECT_ROOT
from backend.ingestion import load_and_chunk_directory

PREVIEW_CHARS = 120
SAMPLE_SIZE = 5


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data"
    print(f"Loading PDFs from: {data_dir}")
    print(f"Config: CHUNK_SIZE={CHUNK_SIZE} tokens, CHUNK_OVERLAP={CHUNK_OVERLAP} tokens\n")

    chunks = load_and_chunk_directory(data_dir)
    print(f"Total chunks: {len(chunks)}\n")

    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(c.page_content)) for c in chunks]
    if token_counts:
        print(
            "Token stats -> "
            f"min: {min(token_counts)}, "
            f"median: {median(token_counts):.1f}, "
            f"max: {max(token_counts)}\n"
        )

    for i, chunk in enumerate(chunks[:SAMPLE_SIZE]):
        preview = chunk.page_content[:PREVIEW_CHARS].replace("\n", " ")
        print(f"--- Chunk {i} ---")
        print(f"metadata: {chunk.metadata}")
        print(f"tokens: {len(enc.encode(chunk.page_content))}")
        print(f"text: {preview}...")
        print()


if __name__ == "__main__":
    main()
