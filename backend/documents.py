"""Document lifecycle: ingest a PDF for a user, list them, delete with cascade.

This module is the join between the two stores. SQLite owns the ``documents``
row; Chroma owns the vectors, stamped with the row's ``document_id``. Keeping
both sides of every mutation in one place is what stops them drifting apart --
the failure mode being a deleted document whose chunks are still retrievable and
citable, which is exactly the bug DECISIONS.md B1 was about, one level up.

Deletion order is deliberate: **vectors first, then the row.** If the process
dies between the two steps, the outcome is a row with no vectors -- the document
is unanswerable and still listed, which the user can retry. The reverse order
would leave orphaned vectors with no row to delete them by: still retrievable,
still citable, and no longer visible in the UI to remove.

Per-user files are written under ``data/users/<user_id>/``. Two users may both
upload ``handbook.pdf``, and a shared directory would have one silently
overwrite the other's bytes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import backend.config as config
from backend import db, manifest, semantic
from backend.ingestion import PDFReadError, build_chunks
from backend.markdown_ingestion import extract_document_auto
from backend.user_scope import user_scope
from backend.vectorstore import delete_document_chunks, ingest_documents

logger = logging.getLogger("smartdoc.documents")

DATA_DIR = config.PROJECT_ROOT / "data"
USER_DATA_DIR = DATA_DIR / "users"


def user_upload_dir(user_id: str) -> Path:
    """Per-user upload directory, created on demand."""
    path = USER_DATA_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def stored_path(user_id: str, filename: str) -> Path:
    """Where this user's copy of ``filename`` lives on disk."""
    return user_upload_dir(user_id) / Path(filename).name


def ingest_pdf_for_user(user_id: str, filename: str, content: bytes) -> dict:
    """Save, parse, chunk, embed, and index ``content`` as this user's document.

    Re-uploading a filename reuses the existing ``documents`` row, so the
    ``document_id`` stamped on the new chunks matches the one on the old chunks
    that ``ingest_documents`` is about to replace.

    Returns:
        ``{"document_id", "filename", "pages_parsed", "chunks_created",
        "chunks_indexed", "extraction_mode"}``.
    """
    dest = stored_path(user_id, filename)
    dest.write_bytes(content)

    record = db.upsert_document(
        user_id=user_id, filename=filename, size_bytes=len(content)
    )

    # V3.1: markdown-first when the flag is on, plain text otherwise and as the
    # fallback. ``user_id`` is passed so the generated markdown is cached under
    # this owner and never in a location another account reads.
    parsed = extract_document_auto(dest, user_id=user_id)
    parents, children = build_chunks(parsed)
    if not children:
        raise PDFReadError(
            f"No extractable text in '{filename}'. Scanned PDFs need OCR before "
            "they can be indexed."
        )

    db.set_document_extraction(
        user_id, record["id"], parsed.extraction_mode, parsed.markdown_path
    )

    # V3.3 Layer B, then Layer C. Order matters: the manifest aggregates the topic
    # and entity lists Layer B produces, so labelling has to finish first.
    semantic.annotate(children)
    manifest_items = manifest.store_manifest(
        user_id, record["id"], manifest.build_manifest(parsed, children)
    )

    # The scope is set here as well as by the request dependency: ingestion also
    # runs from scripts, and the write path must never be able to land chunks
    # with no owner.
    with user_scope(user_id):
        indexed = ingest_documents(
            children, parents=parents, document_id=record["id"]
        )

    return {
        "document_id": record["id"],
        "filename": filename,
        "pages_parsed": parsed.page_count,
        "chunks_created": len(children),
        "chunks_indexed": indexed,
        # Surfaced so a document that degraded to the fallback path is visible at
        # upload time, not only by inspecting the row later.
        "extraction_mode": parsed.extraction_mode,
        # V3.3: how many enumerable items this document is now known to contain.
        "manifest_items": manifest_items,
    }


def _resolve_size(user_id: str, record: dict) -> int | None:
    """Fill in a missing ``size_bytes`` from the stored file, if it is still there.

    Rows written before the column existed -- including the whole pre-V2 corpus
    adopted by ``scripts/seed_dev_user.py`` -- have NULL. Reading the file's real
    size is the honest answer; reporting 0 would show a 33-page manual as taking
    no space. When the file is genuinely gone the value stays None, and the API
    reports it as unknown rather than as zero.
    """
    if record.get("size_bytes") is not None:
        return record["size_bytes"]

    # Legacy documents were ingested from data/ directly, before per-user
    # directories existed, so both locations are worth checking.
    for candidate in (
        stored_path(user_id, record["filename"]),
        DATA_DIR / Path(record["filename"]).name,
    ):
        try:
            if candidate.is_file():
                return candidate.stat().st_size
        except OSError:  # pragma: no cover - unreadable path is just "unknown"
            continue
    return None


def list_documents_for_user(user_id: str) -> list[dict]:
    """This user's documents, each with a resolved size.

    Ownership is in the query, not a later filter.
    """
    return [
        {**record, "size_bytes": _resolve_size(user_id, record)}
        for record in db.list_documents(user_id)
    ]


def delete_document_for_user(user_id: str, document_id: str) -> dict | None:
    """Delete a document the user owns: chunks, parents, file, then the row.

    Returns a summary, or None if ``document_id`` is not this user's document --
    which the endpoint reports as 404, so the response cannot be used to probe
    whether another user's document id exists.
    """
    record = db.get_document(user_id, document_id)
    if record is None:
        return None

    with user_scope(user_id):
        chunks_deleted = delete_document_chunks(document_id)

    path = stored_path(user_id, record["filename"])
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # The indexed copy is gone, which is what makes the document
        # unanswerable; a stranded file on disk is not worth failing the call.
        logger.warning("Could not remove stored file %s", path, exc_info=True)

    db.delete_document_row(user_id, document_id)

    return {
        "document_id": document_id,
        "filename": record["filename"],
        "chunks_deleted": chunks_deleted,
    }
