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

Ownership (V2)
--------------
Chunks written with no owner are visible to NOBODY once ``MULTI_USER_ENABLED``
is on -- every request-time read is filtered by the signed-in user. Pass
``--user`` to index a corpus into a real account; without it the run stays
unscoped (the V1 behaviour the evaluation harness relies on) and says so.

Usage:
    .venv/bin/python -m backend.ingest [--reset] [--force] [--data-dir DIR]
    .venv/bin/python -m backend.ingest --user dev@smartdoc.local
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import backend.config as config
from backend import db
from backend.ingestion import _pdfs_in, build_chunks, extract_document
from backend.user_scope import user_scope
from backend.vectorstore import (
    collection_stats,
    get_collection,
    indexed_hashes,
    ingest_documents,
    reset_collection,
)


def _resolve_owner(identifier: str | None) -> dict | None:
    """Look up the account named by ``--user``, or None for an unscoped run."""
    if not identifier:
        if config.MULTI_USER_ENABLED:
            print(
                "WARNING: no --user given. Chunks will be written without an "
                "owner and will be invisible to every signed-in user. This is "
                "the right mode for the evaluation harness and the wrong one "
                "for populating an account.\n"
            )
        return None
    user = db.get_user_by_email(identifier) or db.get_user_by_id(identifier)
    if user is None:
        raise SystemExit(
            f"No account matches {identifier!r}. Create one via POST /auth/signup, "
            "or run `python -m scripts.seed_dev_user` for the dev account."
        )
    return user


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
    parser.add_argument(
        "--user",
        default=None,
        help=(
            "Email or id of the account to index into. Omit for an unscoped "
            "run: chunks are written with no owner and no signed-in user can "
            "see them."
        ),
    )
    args = parser.parse_args()

    owner = _resolve_owner(args.user)

    print(f"Data directory : {args.data_dir}")
    print(f"Owner          : {owner['email'] if owner else 'NONE (unscoped)'}")
    print(f"Collection     : {config.CHROMA_COLLECTION} at {config.CHROMA_DIR}")
    print(f"Embedding model: {config.EMBED_MODEL}")
    print(
        f"Chunking       : child {config.CHILD_CHUNK_SIZE}/"
        f"{config.CHILD_CHUNK_OVERLAP} tokens, parent {config.PARENT_CHUNK_SIZE}\n"
    )

    # Bind the scope around the whole run, so the content-hash check, the
    # replacement delete, and the writes all agree on whose corpus this is.
    with user_scope(owner["id"] if owner else None):
        _run(args, owner)


def _run(args, owner: dict | None) -> None:
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

        document_id = (
            db.upsert_document(owner["id"], parsed.source)["id"] if owner else ""
        )
        indexed = ingest_documents(
            children, parents=parents, document_id=document_id
        )
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
