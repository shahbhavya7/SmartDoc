"""SmartDoc FastAPI application.

Phase 0: health check only. RAG endpoints arrive in later phases.

Run with:
    uvicorn backend.main:app --reload
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="SmartDoc API",
    description="RAG-based document Q&A assistant.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness probe used by the Streamlit client and by deploy checks."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
