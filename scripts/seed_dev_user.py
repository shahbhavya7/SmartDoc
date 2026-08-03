"""Create the development account and adopt the pre-V2 corpus into it.

Everything indexed before V2 has no owner, and under the new rules an unowned
chunk is invisible to everyone. Rather than re-embedding the corpus (or deleting
it), this backfill stamps ``user_id`` and ``document_id`` onto the existing
chunk metadata and registers one ``documents`` row per source, so the whole
existing library becomes the dev user's library and stays queryable.

The backfill updates metadata only. Vectors, ids, chunk text, chunk boundaries,
and the parent sidecar's contents are untouched, so no embedding calls are made
and retrieval behaves exactly as before -- for this one user.

Legacy chunk ids are NOT rewritten. Their ``prev_id``/``next_id``/``parent_id``
links are unprefixed and point at each other consistently, so renaming the ids
would break every link for no benefit. New uploads get namespaced ids; the two
schemes coexist because lookups resolve by metadata, not by id shape.

Idempotent: safe to re-run.

Usage:
    python -m scripts.seed_dev_user
    python -m scripts.seed_dev_user --password 'something-else'
"""

from __future__ import annotations

import argparse
import sys

import backend.config as config
from backend import auth, db
from backend.user_scope import user_scope
from backend.vectorstore import (
    _load_parents,
    _write_parents,
    get_collection,
)

CHROMA_UPDATE_BATCH = 200


def ensure_dev_user(password: str) -> dict:
    """Get-or-create the dev account with a stable, predictable id."""
    existing = db.get_user_by_email(config.DEV_USER_EMAIL)
    if existing:
        return existing
    auth.validate_password(password)
    return db.create_user(
        email=config.DEV_USER_EMAIL,
        password_hash=auth.hash_password(password),
        user_id=config.DEV_USER_ID,
    )


def backfill_chunks(user_id: str) -> dict:
    """Assign every ownerless chunk to ``user_id``; return a per-source summary.

    Runs unscoped on purpose -- it must see the chunks that no scope can see,
    which is precisely why it is a maintenance script and not an endpoint.
    """
    collection = get_collection()
    got = collection.get(include=["metadatas"])
    ids = got.get("ids") or []
    metadatas = got.get("metadatas") or []

    # One documents row per distinct source, created before the chunks are
    # stamped so every chunk gets a document_id that resolves.
    orphan_sources = sorted(
        {
            (meta or {}).get("source")
            for cid, meta in zip(ids, metadatas)
            if (meta or {}).get("source") and not (meta or {}).get("user_id")
        }
    )
    document_ids = {
        source: db.upsert_document(user_id=user_id, filename=source)["id"]
        for source in orphan_sources
    }

    pending_ids: list[str] = []
    pending_meta: list[dict] = []
    per_source: dict[str, int] = {}

    for cid, meta in zip(ids, metadatas):
        meta = dict(meta or {})
        if meta.get("user_id"):
            continue
        source = meta.get("source")
        if not source:
            continue
        meta["user_id"] = user_id
        meta["document_id"] = document_ids[source]
        pending_ids.append(cid)
        pending_meta.append(meta)
        per_source[source] = per_source.get(source, 0) + 1

    for start in range(0, len(pending_ids), CHROMA_UPDATE_BATCH):
        collection.update(
            ids=pending_ids[start : start + CHROMA_UPDATE_BATCH],
            metadatas=pending_meta[start : start + CHROMA_UPDATE_BATCH],
        )

    return {
        "chunks_updated": len(pending_ids),
        "per_source": per_source,
        "document_ids": document_ids,
    }


def backfill_parents(user_id: str, document_ids: dict[str, str]) -> int:
    """Stamp ownership onto parent records so parent expansion still resolves.

    Without this the parents remain ownerless, ``get_parents`` filters them all
    out, and every answer silently degrades to child-only context -- working, but
    quietly worse.
    """
    store = _load_parents()
    updated = 0
    for record in store.values():
        if record.get("user_id"):
            continue
        record["user_id"] = user_id
        source = record.get("source")
        if source in document_ids:
            record["document_id"] = document_ids[source]
        updated += 1
    if updated:
        _write_parents(store)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--password",
        default=config.DEV_USER_PASSWORD,
        help="Password for the dev account (default: DEV_USER_PASSWORD from .env).",
    )
    args = parser.parse_args()

    db.init_db()
    user = ensure_dev_user(args.password)
    print(f"Dev user: {user['email']}  (id {user['id']})")

    with user_scope(None):  # explicit: the backfill must run unscoped
        chunks = backfill_chunks(user["id"])
        parents = backfill_parents(user["id"], chunks["document_ids"])

    if chunks["chunks_updated"]:
        print(f"\nAdopted {chunks['chunks_updated']} previously ownerless chunks:")
        for source, count in sorted(chunks["per_source"].items()):
            print(f"  {count:>5}  {source}")
        print(f"Parent records stamped: {parents}")
    else:
        print("\nNo ownerless chunks found -- nothing to adopt.")

    print("\nSign in with:")
    print(f"  email:    {user['email']}")
    print(f"  password: {args.password}")
    print(
        "\n  curl -s -X POST http://127.0.0.1:8000/auth/login "
        "-H 'Content-Type: application/json' \\\n"
        f"       -d '{{\"email\":\"{user['email']}\",\"password\":\"{args.password}\"}}'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
