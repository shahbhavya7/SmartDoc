"""Plumbing verification ONLY -- proves persistence/idempotency/reload logic
using a deterministic fake embedder, since no OPENAI_API_KEY is configured.

This never touches the real chroma_store/: it writes to a directory given on
the command line (intended to be a scratch/temp dir). It is not part of the
production ingest path -- backend/vectorstore.py and backend/ingest.py only
ever use the real OpenAI embedder and raise a clear error if the key is
missing.

Subcommands:
    ingest <dir>   -- chunk data/ and upsert into <dir>/chroma_store (fake embed)
    stats <dir>    -- print collection count at <dir>/chroma_store
    query <dir> "<question>"  -- query <dir>/chroma_store (fake embed), print hits

Usage:
    .venv/bin/python -m scripts.verify_persistence_fake ingest /tmp/smartdoc_verify
    .venv/bin/python -m scripts.verify_persistence_fake stats /tmp/smartdoc_verify
    .venv/bin/python -m scripts.verify_persistence_fake query /tmp/smartdoc_verify "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.config import PROJECT_ROOT
from backend.ingestion import load_and_chunk_directory
from backend.vectorstore import collection_stats, ingest_documents, query_collection
from scripts.eval_chunk_size import _fake_embed_fn


def cmd_ingest(persist_dir: Path) -> None:
    chunks = load_and_chunk_directory(PROJECT_ROOT / "data")
    n = ingest_documents(
        chunks,
        collection_name="smartdoc_verify",
        persist_directory=persist_dir,
        reset=False,
        embed_fn=_fake_embed_fn,
    )
    print(f"Upserted {n} chunks (fake embedder) into {persist_dir}")
    print(collection_stats(collection_name="smartdoc_verify", persist_directory=persist_dir))


def cmd_stats(persist_dir: Path) -> None:
    print(collection_stats(collection_name="smartdoc_verify", persist_directory=persist_dir))


def cmd_query(persist_dir: Path, question: str) -> None:
    hits = query_collection(
        question,
        collection_name="smartdoc_verify",
        persist_directory=persist_dir,
        embed_fn=_fake_embed_fn,
    )
    for h in hits:
        print(f"id={h['id']} metadata={h['metadata']} distance={h['distance']:.4f}")
        print(f"  text: {h['document'][:100]!r}")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    command, dir_arg = sys.argv[1], sys.argv[2]
    persist_dir = Path(dir_arg) / "chroma_store"

    if command == "ingest":
        cmd_ingest(persist_dir)
    elif command == "stats":
        cmd_stats(persist_dir)
    elif command == "query":
        if len(sys.argv) < 4:
            print("query requires a question argument")
            sys.exit(1)
        cmd_query(persist_dir, sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
