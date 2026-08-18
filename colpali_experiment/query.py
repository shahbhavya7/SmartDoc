"""Query the ColPali store: score a text question against every stored page
and return the top pages. In-memory, full-scan -- fine at experimental scale
(brief item 5), not a production index.
"""

from __future__ import annotations

import torch

from colpali_experiment import store
from colpali_experiment.embedder import embed_queries, score


def top_pages(user_id: str, question: str, top_k: int = 5) -> list[dict]:
    """Rank this user's stored pages against ``question`` by MaxSim score."""
    rows = store.get_user_embeddings(user_id)
    if not rows:
        return []

    query_embeddings = embed_queries([question])
    page_embeddings = [torch.from_numpy(r["embedding"]) for r in rows]
    scores = score(query_embeddings, page_embeddings)  # shape (1, n_pages)
    ranked = sorted(
        zip(rows, scores[0].tolist()), key=lambda pair: pair[1], reverse=True
    )
    return [
        {
            "document_id": r["document_id"],
            "filename": r["filename"],
            "page_number": r["page_number"],
            "score": s,
        }
        for r, s in ranked[:top_k]
    ]
