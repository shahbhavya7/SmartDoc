"""SmartDoc FastAPI application.

HTTP layer over the RAG pipeline. This module holds NO retrieval or generation
logic of its own -- it validates input, calls the ``backend`` modules, and maps
their exceptions to structured JSON, so the whole system is drivable via curl or
the Swagger UI at ``/docs``.

Endpoints:
    GET  /health  -- liveness probe, enriched with collection stats.
    POST /upload  -- ingest one or more PDFs; per-file success/error so one bad
                     file does not fail the batch.
    POST /ask     -- answer a question via ``backend.rag.query``.

Hardening decisions:
    - Blank/whitespace-only question -> HTTP 400.
    - Questions over ``config.MAX_QUESTION_CHARS`` -> HTTP 400, REJECTED rather
      than silently truncated. Truncating can change the meaning of a question
      and produce a confidently wrong answer with no sign anything was altered.
    - Non-English questions are not blocked; the pipeline translates them for
      retrieval and answers in the asked language.
    - Any OpenAI failure -> HTTP 502, clean JSON, no stack trace, server stays up.
    - Upload filenames are sanitised to a bare basename (defeating traversal),
      only .pdf is accepted, and bodies are read in bounded chunks with a hard
      size cap so a huge upload cannot exhaust memory.
    - Every error response shares one shape: ``{"error": {"type", "message"}}``.

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import openai
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import MAX_QUESTION_CHARS, MAX_UPLOAD_MB, PROJECT_ROOT
from backend.ingestion import PDFReadError, build_chunks, extract_document
from backend.rag import GenerationError, InvalidQuestionError, RagError, query
from backend.vectorstore import VectorStoreError, collection_stats, ingest_documents

logger = logging.getLogger("smartdoc.api")

DATA_DIR = PROJECT_ROOT / "data"
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_READ_CHUNK = 1024 * 1024  # stream uploads in 1 MiB chunks

app = FastAPI(
    title="SmartDoc API",
    description=(
        "RAG-based document Q&A assistant. Upload company PDFs, then ask "
        "plain-English questions and receive answers grounded in, and cited "
        "from, the ingested documents. Retrieval adapts to the question type: "
        "fact lookup, comparison, multi-step, procedural, document synthesis, "
        "exhaustive extraction, or cross-document reasoning."
    ),
    version="0.5.0",
)

# Permissive CORS for a Streamlit client on a different localhost port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# API-layer-only exceptions
# --------------------------------------------------------------------------


class QuestionTooLongError(Exception):
    """Raised when a submitted question exceeds ``MAX_QUESTION_CHARS``."""


class UploadValidationError(Exception):
    """Raised for a structurally invalid upload (bad filename/type/size)."""


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness probe response. ``status`` is the stable contract field."""

    status: Literal["ok"]
    collection: str | None = Field(default=None)
    indexed_chunks: int | None = Field(default=None)
    documents: int | None = Field(default=None)
    embedding_model: str | None = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "collection": "smartdoc",
                "indexed_chunks": 471,
                "documents": 11,
                "embedding_model": "text-embedding-3-small",
            }
        }
    }


class AskRequest(BaseModel):
    """Body for ``POST /ask``."""

    question: str = Field(
        ...,
        description=(
            "A plain-English question about the ingested documents. Must not be "
            f"blank, and at most {MAX_QUESTION_CHARS} characters."
        ),
        examples=["How many days of annual leave do I get?"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {"question": "How many days of annual leave do I get?"}
        }
    }


class SourceModel(BaseModel):
    """A single structural citation."""

    source: str = Field(..., description="Source PDF filename.")
    page: int = Field(..., description="1-indexed page number.")
    snippet: str = Field(..., description="Excerpt of the cited passage.")
    section: str = Field(default="", description="Nearest enclosing heading.")
    page_end: int | None = Field(
        default=None, description="Last page when the passage spans a break."
    )


class GroundingModel(BaseModel):
    """Verification verdict attached to every answered response."""

    checked: bool
    faithful: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    unverified_numbers: list[str] = Field(
        default_factory=list,
        description=(
            "Figures not found verbatim in context. Informational: a "
            "legitimately derived value lands here too."
        ),
    )
    note: str = ""
    repaired: str = Field(
        default="",
        description="Remediation applied: regenerated, pruned, declined, or none.",
    )
    removed_claims: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Response body for ``POST /ask``."""

    answer: str
    sources: list[SourceModel]
    query_type: str = Field(default="", description="Detected intent.")
    grounding: GroundingModel | None = None
    diagnostics: dict | None = Field(
        default=None, description="Routing, retrieval, and latency diagnostics."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Standard band employees accrue 20 days of annual leave; Executive band employees accrue 28.",
                "sources": [
                    {
                        "source": "employee_handbook.pdf",
                        "page": 4,
                        "snippet": "Employees in the Standard band accrue twenty (20) days...",
                        "section": "3. Annual Leave Entitlement",
                        "page_end": None,
                    }
                ],
                "query_type": "comparison",
                "grounding": {
                    "checked": True,
                    "faithful": True,
                    "unsupported_claims": [],
                    "unverified_numbers": [],
                    "note": "",
                    "repaired": "",
                    "removed_claims": [],
                },
            }
        }
    }


class UploadFileResult(BaseModel):
    """Per-file outcome for one item in a ``POST /upload`` batch."""

    filename: str
    status: Literal["success", "error"]
    pages_parsed: int | None = None
    chunks_created: int | None = None
    chunks_indexed: int | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    """Response body for ``POST /upload``: a per-file batch summary."""

    files: list[UploadFileResult]
    total_chunks_indexed: int
    collection_name: str
    collection_count: int


class ErrorDetail(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    """Every error path returns this one shape."""

    error: ErrorDetail


# --------------------------------------------------------------------------
# Exception handlers
# --------------------------------------------------------------------------


def _error_body(error_type: str, message: str) -> dict:
    return {"error": {"type": error_type, "message": message}}


@app.exception_handler(InvalidQuestionError)
async def _invalid_question_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=400, content=_error_body("invalid_question", str(exc))
    )


@app.exception_handler(QuestionTooLongError)
async def _question_too_long_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=400, content=_error_body("question_too_long", str(exc))
    )


@app.exception_handler(GenerationError)
async def _generation_error_handler(request: Request, exc) -> JSONResponse:
    logger.error("OpenAI generation call failed: %s", exc)
    return JSONResponse(
        status_code=502,
        content=_error_body(
            "generation_failed",
            "The language model service is currently unavailable or "
            "misconfigured. Please try again later.",
        ),
    )


@app.exception_handler(PDFReadError)
async def _pdf_read_error_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body("pdf_read_error", str(exc)))


@app.exception_handler(UploadValidationError)
async def _upload_validation_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(status_code=400, content=_error_body("invalid_upload", str(exc)))


@app.exception_handler(VectorStoreError)
async def _vector_store_error_handler(request: Request, exc) -> JSONResponse:
    logger.error("Vector store error: %s", exc)
    return JSONResponse(
        status_code=500, content=_error_body("vector_store_error", str(exc))
    )


@app.exception_handler(openai.OpenAIError)
async def _openai_error_handler(request: Request, exc) -> JSONResponse:
    # Catches OpenAI SDK errors raised OUTSIDE backend.rag's own try/except --
    # e.g. an embedding-call auth failure during retrieval or upload -- so a
    # bad key at ANY call site returns the same clean, non-crashing error.
    logger.error("OpenAI API call failed: %s", exc)
    return JSONResponse(
        status_code=502,
        content=_error_body(
            "upstream_api_error",
            "The OpenAI API is currently unavailable or the configured API key "
            "is invalid. Please verify OPENAI_API_KEY and try again.",
        ),
    )


@app.exception_handler(RagError)
async def _rag_error_handler(request: Request, exc) -> JSONResponse:
    logger.error("Unhandled RAG error: %s", exc)
    return JSONResponse(
        status_code=500, content=_error_body("rag_error", "The query could not be completed.")
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Replace FastAPI's default {"detail": [...]} dump with our structured shape.
    logger.info("Request validation failed for %s: %s", request.url.path, exc.errors())
    messages = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []) if part != "body")
        messages.append(f"{loc}: {err.get('msg')}" if loc else err.get("msg", "invalid"))
    return JSONResponse(
        status_code=400,
        content=_error_body("validation_error", "; ".join(messages) or "Invalid body."),
    )


@app.exception_handler(Exception)
async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last-resort guard: never let an unexpected exception crash the server or
    # leak a stack trace to the client.
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_body("internal_error", "An unexpected error occurred."),
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Confirms the API is up and reports collection stats.",
)
def health() -> HealthResponse:
    """Liveness probe used by clients and deploy checks."""
    try:
        stats = collection_stats()
        return HealthResponse(
            status="ok",
            collection=stats["name"],
            indexed_chunks=stats["count"],
            documents=stats.get("documents"),
            embedding_model=stats.get("embedding_model"),
        )
    except Exception:  # pragma: no cover - health must never fail on store issues
        logger.warning("collection_stats() failed during /health.", exc_info=True)
        return HealthResponse(status="ok")


def _sanitize_filename(raw_name: str | None) -> str:
    """Reduce a client-supplied filename to a safe basename ending in .pdf."""
    if not raw_name:
        raise UploadValidationError("Uploaded file is missing a filename.")
    base = Path(raw_name).name  # drops directory components / traversal
    if not base or base in (".", ".."):
        raise UploadValidationError(f"Invalid filename: {raw_name!r}")
    if Path(base).suffix.lower() != ".pdf":
        raise UploadValidationError(
            f"Rejected '{raw_name}': only .pdf files are accepted."
        )
    return base


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read ``upload`` in bounded chunks, aborting once ``max_bytes`` is passed."""
    buffer = bytearray()
    while True:
        chunk = await upload.read(UPLOAD_READ_CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise UploadValidationError(
                f"File exceeds the maximum upload size of {MAX_UPLOAD_MB} MB."
            )
    return bytes(buffer)


def _ingest_one_file(filename: str, content: bytes) -> UploadFileResult:
    """Write ``content`` to the data dir under ``filename`` and index it.

    Uses the structure-aware pipeline and REPLACES any previously indexed
    version of this filename -- re-uploading an edited document that produces
    fewer chunks would otherwise leave the surplus old chunks live and citable.
    """
    dest = DATA_DIR / filename
    dest.write_bytes(content)

    parsed = extract_document(dest)
    parents, children = build_chunks(parsed)
    if not children:
        raise PDFReadError(
            f"No extractable text in '{filename}'. Scanned PDFs need OCR before "
            "they can be indexed."
        )
    indexed = ingest_documents(children, parents=parents)

    return UploadFileResult(
        filename=filename,
        status="success",
        pages_parsed=parsed.page_count,
        chunks_created=len(children),
        chunks_indexed=indexed,
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index one or more PDFs",
    description=(
        "Accepts one or more PDFs (multipart/form-data). Each is saved, parsed, "
        "chunked, embedded, and indexed. Re-uploading the same filename REPLACES "
        "its chunks. One invalid file does not fail the rest of the batch -- "
        "check each entry's `status`."
    ),
)
async def upload(
    files: list[UploadFile] = File(..., description="One or more .pdf files."),
) -> UploadResponse:
    """Ingest, embed, and index each uploaded PDF; return a per-file summary."""
    results: list[UploadFileResult] = []

    for upload_file in files:
        original_name = upload_file.filename or "<unnamed>"
        try:
            content_type = (upload_file.content_type or "").lower()
            if content_type and content_type not in (
                "application/pdf",
                "application/octet-stream",
            ):
                raise UploadValidationError(
                    f"Rejected '{original_name}': unsupported content type "
                    f"'{content_type}', expected application/pdf."
                )

            safe_name = _sanitize_filename(original_name)
            content = await _read_limited(upload_file, MAX_UPLOAD_BYTES)
            if not content:
                raise UploadValidationError(f"'{original_name}' is empty.")

            result = _ingest_one_file(safe_name, content)
        except UploadValidationError as exc:
            result = UploadFileResult(
                filename=original_name, status="error", error=str(exc)
            )
        except PDFReadError as exc:
            result = UploadFileResult(
                filename=original_name, status="error", error=str(exc)
            )
        except VectorStoreError as exc:
            logger.error("Vector store error ingesting %s: %s", original_name, exc)
            result = UploadFileResult(
                filename=original_name, status="error", error=str(exc)
            )
        except openai.OpenAIError as exc:
            logger.error("OpenAI error embedding %s: %s", original_name, exc)
            result = UploadFileResult(
                filename=original_name,
                status="error",
                error="Embedding failed: the OpenAI API is unavailable or the key is invalid.",
            )
        except Exception:  # noqa: BLE001 - one bad file must not fail the batch
            logger.exception("Unexpected error ingesting %s", original_name)
            result = UploadFileResult(
                filename=original_name,
                status="error",
                error="Unexpected error while processing this file.",
            )
        finally:
            await upload_file.close()

        results.append(result)

    stats = collection_stats()
    return UploadResponse(
        files=results,
        total_chunks_indexed=sum(
            r.chunks_indexed or 0 for r in results if r.status == "success"
        ),
        collection_name=stats["name"],
        collection_count=stats["count"],
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question grounded in the ingested documents",
    description=(
        "Runs intent-aware retrieval-augmented generation over the persisted "
        "collection and answers using ONLY retrieved context. If the documents "
        "do not contain the answer, responds with a fixed refusal and an empty "
        "`sources` list instead of guessing. Answers whose claims fail "
        "verification are repaired or withdrawn, never returned as-is. "
        f"Questions over {MAX_QUESTION_CHARS} characters are rejected."
    ),
)
def ask(payload: AskRequest) -> AskResponse:
    """Answer ``payload.question`` via the RAG pipeline."""
    question = payload.question
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionTooLongError(
            f"Question is {len(question)} characters, which exceeds the "
            f"{MAX_QUESTION_CHARS}-character limit. Please shorten it."
        )
    # Blank questions raise InvalidQuestionError inside query() itself.
    return AskResponse(**query(question).to_dict())


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
