"""A distinct FastAPI router, mounted at /colpali/*.

Deliberately NOT folded into the existing /upload endpoint (the brief's hard
constraint): this router only ever operates on documents that already exist
in the hybrid pipeline, triggered separately. It reuses the same auth
dependency as every other endpoint so it is scoped to the signed-in user, but
it has no effect whatsoever on ``backend.rag.query`` or any hybrid-path
behaviour -- it reads ``backend.db`` document rows and writes only to this
experiment's own SQLite file.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user_id
from colpali_experiment import ingest, query as colpali_query, store

router = APIRouter(prefix="/colpali", tags=["colpali-experiment"])


@router.post("/ingest")
def colpali_ingest(
    document_id: str | None = None,
    force: bool = False,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Embed one (or, if omitted, every) existing document this user owns."""
    if document_id:
        try:
            return ingest.ingest_document(document_id, user_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    results = ingest.ingest_all_for_user(user_id, skip_existing=not force)
    return {"results": results}


@router.get("/status")
def colpali_status(user_id: str = Depends(get_current_user_id)) -> dict:
    """Inspect what has been embedded for this user -- page counts, no vectors."""
    rows = store.get_user_embeddings(user_id)
    by_doc: dict[str, dict] = {}
    for r in rows:
        entry = by_doc.setdefault(
            r["document_id"], {"filename": r["filename"], "pages": 0}
        )
        entry["pages"] += 1
    return {"documents": by_doc, "total_pages": len(rows)}


@router.get("/query")
def colpali_ask(
    question: str, top_k: int = 5, user_id: str = Depends(get_current_user_id)
) -> dict:
    """MaxSim-rank this user's stored pages against ``question``. Experimental
    inspection endpoint -- does not generate an answer, only shows retrieval.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be blank")
    return {"results": colpali_query.top_pages(user_id, question, top_k=top_k)}
