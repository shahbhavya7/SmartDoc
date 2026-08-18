"""ColPali ingestion: reuse existing document rows, add only embedding artifacts.

Runs entirely alongside the hybrid pipeline's ``backend.documents.ingest_pdf_for_user``.
It never calls that function and never touches ``smartdoc.db`` or
``chroma_store/`` -- it only READS the existing ``documents`` table (for
``document_id``, ``user_id``, ``filename``) and the file already stored on
disk by the hybrid path, then WRITES to this experiment's own store.
"""

from __future__ import annotations

import logging
from pathlib import Path

from colpali_experiment import config, store
from colpali_experiment.embedder import device_in_use, embed_page_images
from colpali_experiment.renderer import render_pdf_pages

from backend import db as backend_db
from backend.documents import DATA_DIR, stored_path

logger = logging.getLogger("colpali_experiment.ingest")


def _resolve_pdf_path(user_id: str, filename: str) -> Path | None:
    """Same two locations backend.documents._resolve_size checks -- per-user
    dir first, then the legacy flat data/ dir for pre-V2 adopted documents.
    """
    for candidate in (stored_path(user_id, filename), DATA_DIR / Path(filename).name):
        if candidate.is_file():
            return candidate
    return None


def ingest_document(
    document_id: str, user_id: str, batch_size: int = 1, max_pages: int | None = None
) -> dict:
    """Render + embed one EXISTING document's pages, storing embeddings only.

    ``batch_size`` defaults to 1 -- one page embedded at a time, deliberately,
    so a run on modest hardware (CPU, no dedicated GPU) never holds more than
    one page's activations in memory at once. ``max_pages`` caps how many
    pages are embedded at all, for a cheap proof-of-concept run against a
    handful of pages rather than a whole document.

    Returns a summary dict; raises FileNotFoundError if the record's PDF is
    not on disk (never touches the ``documents`` row in that case).
    """
    record = backend_db.get_document(user_id, document_id)
    if record is None:
        raise ValueError(f"No document {document_id!r} owned by {user_id!r}.")

    filename = record["filename"]
    pdf_path = _resolve_pdf_path(user_id, filename)
    if pdf_path is None:
        raise FileNotFoundError(f"Stored PDF for {filename!r} not found on disk.")

    pages = render_pdf_pages(pdf_path, document_id)
    if max_pages is not None:
        pages = pages[:max_pages]
    if not pages:
        return {"document_id": document_id, "filename": filename, "pages": 0}

    total_patches = 0
    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        images = [img for _, img in batch]
        embeddings = embed_page_images(images)
        for (page_number, _), embedding in zip(batch, embeddings):
            arr = embedding.numpy()
            store.replace_page_embedding(
                document_id=document_id,
                user_id=user_id,
                filename=filename,
                page_number=page_number,
                embedding=arr,
                model_name=config.COLPALI_MODEL_NAME,
            )
            total_patches += arr.shape[0]
        logger.info(
            "colpali ingest: %s pages %d-%d/%d embedded",
            filename, start + 1, start + len(batch), len(pages),
        )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "total_patches": total_patches,
        "device": device_in_use(),
        "model": config.COLPALI_MODEL_NAME,
    }


def ingest_all_for_user(user_id: str, skip_existing: bool = True) -> list[dict]:
    """Ingest every document this user already owns in the hybrid pipeline.

    ``skip_existing`` avoids re-embedding a document already stored under the
    current model name -- cheap re-runs while iterating on this experiment.
    """
    results = []
    for record in backend_db.list_documents(user_id):
        document_id = record["id"]
        if skip_existing and store.get_document_embeddings(document_id):
            existing = store.get_document_embeddings(document_id)
            if existing and existing[0]["model_name"] == config.COLPALI_MODEL_NAME:
                results.append(
                    {
                        "document_id": document_id,
                        "filename": record["filename"],
                        "skipped": True,
                        "pages": len(existing),
                    }
                )
                continue
        try:
            results.append(ingest_document(document_id, user_id))
        except FileNotFoundError as exc:
            logger.warning("colpali ingest skipped %s: %s", record["filename"], exc)
            results.append(
                {"document_id": document_id, "filename": record["filename"], "error": str(exc)}
            )
    return results
