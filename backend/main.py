"""SmartDoc FastAPI application.

HTTP layer over the RAG pipeline. This module holds NO retrieval or generation
logic of its own -- it validates input, calls the ``backend`` modules, and maps
their exceptions to structured JSON, so the whole system is drivable via curl or
the Swagger UI at ``/docs``.

Endpoints:
    GET  /health          -- liveness probe (public).
    POST /auth/signup     -- create an email/password account, returns a JWT.
    POST /auth/login      -- exchange email/password for a JWT.
    GET  /auth/me         -- the signed-in user.
    GET  /auth/google/login, /auth/google/callback -- Google OAuth, returns a JWT.
    POST /upload          -- ingest one or more PDFs for the signed-in user.
    GET  /documents       -- list the signed-in user's documents.
    DELETE /documents/{id}-- delete a document: SQLite row AND its Chroma chunks.
    POST /ask             -- answer a question over the signed-in user's documents.
    POST/GET /sessions, /sessions/{id}/messages -- chat history, per user.
    GET  /eval/method, /eval/gold-set, /eval/calibration -- how scoring works.
    GET  /eval/runs, /eval/runs/{id} -- past evaluation results, per user.
    POST /eval/test-sets  -- upload a test set to evaluate against.
    POST /eval/runs       -- start an evaluation; GET /eval/jobs/{id} to poll.

Authentication and isolation (V2):
    Every endpoint below ``/health`` and ``/auth`` depends on
    ``get_current_user_id``, which verifies the JWT server-side and returns the
    subject claim. That value is bound as the active scope for the duration of
    the request, and ``backend.vectorstore`` filters every Chroma read by it. No
    endpoint reads a user id from a path, query parameter, or body -- there is no
    such parameter to send.

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
from fastapi import BackgroundTasks, Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

import backend.config as config
from backend import auth, db, documents, evaluation
from backend.auth import (
    AuthError,
    OAuthNotConfigured,
    RegistrationError,
    get_current_user,
    get_current_user_id,
)
from backend.config import MAX_QUESTION_CHARS, MAX_UPLOAD_MB, PROJECT_ROOT
from backend.ingestion import PDFReadError
from backend.memory import (
    answer_meta_question,
    looks_like_meta_question,
    summarize_turn_and_store,
)
from backend.rag import GenerationError, InvalidQuestionError, RagError, query
from backend.user_scope import ScopeError, user_scope
from backend.vectorstore import VectorStoreError, collection_stats, get_chunks_where

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
    version="2.0.0",
)

# CORS for the Next.js client, from an explicit allowlist (CORS_ALLOW_ORIGINS).
# Credentials stay off: the browser sends the JWT in an Authorization header,
# never a cookie, so there is no ambient session for another origin to ride --
# and with credentials off, `allow_origins` is the only thing standing between a
# scraped token and a cross-origin replay, which is why it is no longer "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Signed cookie session, required by authlib to carry the OAuth ``state`` value
# across the redirect to Google and back. It holds nothing else -- API
# authorization is the bearer token, never this cookie.
app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET,
    same_site="lax",
    https_only=False,
)

# Colpali branch experiment: a self-contained router under /colpali/*, additive
# only -- see colpali_experiment/api.py. Guarded by try/except so a missing
# experimental dependency (torch, colpali-engine) cannot take down the hybrid
# app; this is not a code path any other branch or deployment relies on.
#
# The two names below are also used directly by /upload (background fan-out)
# and /ask (RETRIEVAL_BACKEND=colpali dispatch) further down this file. Both
# call sites re-check ``_colpali_available`` before touching them, so a
# missing experimental dependency degrades those call sites too -- /upload's
# hybrid ingestion and /ask's hybrid answers are entirely unaffected either
# way, per the hard isolation rule.
_colpali_available = False
try:
    from colpali_experiment.answer import answer as _colpali_answer
    from colpali_experiment.answer import to_rag_response as _colpali_to_rag_response
    from colpali_experiment.api import router as colpali_router
    from colpali_experiment.ingest import ingest_document_for_upload as _colpali_ingest_for_upload
    from colpali_experiment.store import set_ingest_status as _colpali_set_ingest_status

    app.include_router(colpali_router)
    _colpali_available = True
except ImportError:  # pragma: no cover - experimental dependency not installed
    logger.warning(
        "colpali_experiment not available; /colpali/* routes disabled and "
        "RETRIEVAL_BACKEND=colpali will report an error instead of answering."
    )


@app.on_event("startup")
def _startup() -> None:
    """Create the SQLite schema and refuse an obviously unsafe signing key."""
    db.init_db()
    auth.assert_signing_key_usable()
    # Addendum 2: preload every owner's table vocabulary, so the first question
    # after a restart pays no more than the millionth. Lazy building covers
    # anything this misses (a user created after boot); this only removes the
    # cold-start cost. No-op with PARALLEL_SQL_LOOKUP_ENABLED off.
    from backend import table_store

    warmed = table_store.warm_all()
    if warmed:
        logger.info("Warmed table vocabulary for %d user(s).", warmed)


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
    google_oauth_enabled: bool = Field(
        default=False,
        description=(
            "Whether GOOGLE_CLIENT_ID/SECRET are configured. Lets a sign-in page "
            "hide or disable its Google button instead of sending the user into a "
            "503. Reveals a deployment capability, not user data -- and the login "
            "page it feeds would expose the same fact by having the button at all."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "collection": "smartdoc",
                "indexed_chunks": 471,
                "documents": 11,
                "embedding_model": "text-embedding-3-small",
                "google_oauth_enabled": True,
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
    session_id: str | None = Field(
        default=None,
        description=(
            "A chat session owned by the caller (see POST /sessions). When set, "
            "the question and answer are stored as that session's next turn, the "
            "session's running summary is used to resolve references in the "
            "question, and the summary is updated afterward in the background. "
            "Omit for a stateless, un-remembered question."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {"question": "How many days of annual leave do I get?"}
        }
    }


RetrievalBackend = Literal["hybrid", "colpali"]


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
    session_id: str | None = Field(
        default=None, description="Echoed back when the request named a session."
    )
    backend: RetrievalBackend = Field(
        default="hybrid",
        description=(
            "colpali branch experiment: which pipeline actually answered -- "
            "'hybrid' (the text pipeline described above) or 'colpali' "
            "(colpali_experiment, a separate visual/page-image pipeline). "
            "Selected by RETRIEVAL_BACKEND, overridable per-request via the "
            "`backend` query parameter on this endpoint."
        ),
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
    document_id: str | None = Field(
        default=None, description="SQLite documents.id; use it to delete this document."
    )
    pages_parsed: int | None = None
    chunks_created: int | None = None
    chunks_indexed: int | None = None
    extraction_mode: str | None = Field(
        default=None,
        description=(
            "V3.1: 'markdown', or 'text' when the plain-text path was used -- "
            "either because MARKDOWN_INGESTION_ENABLED is off or because "
            "conversion fell back (a scanned / image-only PDF)."
        ),
    )
    table_cells: int | None = Field(
        default=None,
        description=(
            "Addendum 2: table cells now exactly lookup-able for this document. "
            "0 both when PARALLEL_SQL_LOOKUP_ENABLED is off and when the document "
            "has no tables -- surfaced here for the same reason as "
            "extraction_mode, so a document that produced nothing is visible at "
            "upload time rather than only by inspecting the database."
        ),
    )
    error: str | None = None


class UploadResponse(BaseModel):
    """Response body for ``POST /upload``: a per-file batch summary."""

    files: list[UploadFileResult]
    total_chunks_indexed: int
    collection_name: str
    collection_count: int


class SignupRequest(BaseModel):
    """Body for ``POST /auth/signup``."""

    email: str = Field(..., examples=["alice@example.com"])
    password: str = Field(
        ...,
        description=(
            f"At least {auth.MIN_PASSWORD_CHARS} characters and at most "
            f"{auth.MAX_PASSWORD_BYTES} bytes."
        ),
        examples=["correct-horse-battery"],
    )


class LoginRequest(BaseModel):
    """Body for ``POST /auth/login``."""

    email: str = Field(..., examples=["alice@example.com"])
    password: str = Field(..., examples=["correct-horse-battery"])


class UserModel(BaseModel):
    """A user as returned to the client. Never includes the password hash."""

    id: str
    email: str
    created_at: str
    auth_methods: list[str] = Field(
        default_factory=list, description="'password' and/or 'google'."
    )


class TokenResponse(BaseModel):
    """A successful authentication. ``access_token`` is a signed JWT."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds.")
    user: UserModel


class DocumentModel(BaseModel):
    """One of the signed-in user's documents."""

    id: str
    filename: str
    created_at: str
    chunks: int | None = Field(
        default=None, description="Indexed chunks for this document."
    )
    size_bytes: int | None = Field(
        default=None,
        description=(
            "Size of the stored PDF. None when the size is genuinely unknown "
            "(a pre-V2 row whose file is no longer on disk) -- not zero, which "
            "would misreport it as empty."
        ),
    )
    extraction_mode: str = Field(
        default="text",
        description=(
            "V3.1: 'markdown' when the document was converted to markdown and "
            "split on heading boundaries, 'text' for the plain-text path -- "
            "including a scanned PDF that fell back to it. 'text' is also the "
            "correct value for every row that predates the column."
        ),
    )


class DocumentListResponse(BaseModel):
    documents: list[DocumentModel]
    total_chunks: int = Field(..., description="Indexed chunks across this user only.")
    total_bytes: int = Field(
        default=0,
        description=(
            "Sum of the known document sizes for this user. Documents of unknown "
            "size contribute nothing, so this is a lower bound, and "
            "`documents_with_unknown_size` says how many were left out."
        ),
    )
    documents_with_unknown_size: int = Field(default=0)


class DeleteDocumentResponse(BaseModel):
    """Result of a delete: both stores reported on, so a partial is visible."""

    document_id: str
    filename: str
    chunks_deleted: int
    row_deleted: bool


class SessionCreateRequest(BaseModel):
    title: str = Field(default="", max_length=200)


class SessionModel(BaseModel):
    id: str
    title: str
    created_at: str
    summary: str = Field(
        default="", description="Running conversation summary (see backend/memory.py)."
    )
    last_document: str | None = Field(
        default=None, description="Filename last discussed in this session, if any."
    )


class MessageCreateRequest(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class MessageModel(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: str


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


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc) -> JSONResponse:
    # 401 with WWW-Authenticate so a client can tell "sign in" apart from
    # "you are signed in but this is not yours" (which is a 404 by design).
    return JSONResponse(
        status_code=401,
        content=_error_body("unauthenticated", str(exc)),
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(RegistrationError)
async def _registration_error_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(status_code=400, content=_error_body("registration_failed", str(exc)))


@app.exception_handler(OAuthNotConfigured)
async def _oauth_not_configured_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(status_code=503, content=_error_body("oauth_not_configured", str(exc)))


@app.exception_handler(ScopeError)
async def _scope_error_handler(request: Request, exc) -> JSONResponse:
    # A scoped store operation ran with no user bound. Failing closed with a 500
    # is correct: the alternative is serving an unfiltered result set.
    logger.error("Scope violation on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=_error_body("scope_error", "The request could not be scoped to a user."),
    )


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
    description=(
        "Confirms the API is up. Corpus counts are NOT reported here: /health is "
        "unauthenticated, and a global chunk total would tell an anonymous caller "
        "how much every user has uploaded. Use GET /documents for your own counts."
    ),
)
def health() -> HealthResponse:
    """Liveness probe used by clients and deploy checks."""
    google_enabled = auth.google_configured()
    try:
        stats = collection_stats()
        return HealthResponse(
            status="ok",
            collection=stats["name"],
            embedding_model=stats.get("embedding_model"),
            google_oauth_enabled=google_enabled,
        )
    except Exception:  # pragma: no cover - health must never fail on store issues
        logger.warning("collection_stats() failed during /health.", exc_info=True)
        return HealthResponse(status="ok", google_oauth_enabled=google_enabled)


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def _auth_methods(user: dict) -> list[str]:
    methods = []
    if user.get("password_hash"):
        methods.append("password")
    if user.get("google_sub"):
        methods.append("google")
    return methods


def _to_user_model(user: dict) -> UserModel:
    """Project a user row to the wire shape, dropping the password hash."""
    return UserModel(
        id=user["id"],
        email=user["email"],
        created_at=user["created_at"],
        auth_methods=_auth_methods(user),
    )


def _token_response(user: dict) -> TokenResponse:
    return TokenResponse(
        access_token=auth.create_access_token(user["id"], user["email"]),
        expires_in=config.JWT_EXPIRE_MINUTES * 60,
        user=_to_user_model(user),
    )


@app.post(
    "/auth/signup",
    response_model=TokenResponse,
    status_code=201,
    summary="Create an email/password account",
    description=(
        "Hashes the password with bcrypt and issues a JWT. The token's subject "
        "is the new user's id; every subsequent request is scoped by it."
    ),
)
def signup(payload: SignupRequest) -> TokenResponse:
    return _token_response(auth.signup(payload.email, payload.password))


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Exchange email and password for a JWT",
    description=(
        "Returns the same error for an unknown email and a wrong password, so "
        "the endpoint cannot be used to discover which addresses are registered."
    ),
)
def login(payload: LoginRequest) -> TokenResponse:
    return _token_response(auth.login(payload.email, payload.password))


@app.get(
    "/auth/me",
    response_model=UserModel,
    summary="The signed-in user",
    description="Resolved from the bearer token; takes no parameters.",
)
def whoami(user: dict = Depends(get_current_user)) -> UserModel:
    return _to_user_model(user)


@app.get(
    "/auth/google/login",
    summary="Begin Google sign-in",
    description=(
        "Redirects to Google's consent screen. Returns 503 if "
        "GOOGLE_CLIENT_ID/SECRET are unset -- email/password sign-in still works."
    ),
)
async def google_login(request: Request):
    client = auth.google_client()
    return await client.authorize_redirect(request, config.GOOGLE_REDIRECT_URI)


@app.get(
    "/auth/google/callback",
    summary="Google OAuth callback",
    description=(
        "Exchanges the authorization code for tokens, resolves or creates the "
        "account from the verified ``sub`` claim, and issues the same kind of JWT "
        "that password login issues. With OAUTH_SUCCESS_REDIRECT set, redirects "
        "to the frontend with the token; otherwise returns it as JSON."
    ),
)
async def google_callback(request: Request):
    client = auth.google_client()
    try:
        token = await client.authorize_access_token(request)
    except Exception as exc:  # authlib raises several distinct error types
        logger.warning("Google OAuth exchange failed: %s", exc)
        raise AuthError("Google sign-in failed or was cancelled.") from exc

    claims = token.get("userinfo")
    if not claims:
        # Fall back to the UserInfo endpoint when the id_token was not parsed.
        claims = await client.userinfo(token=token)

    user = auth.upsert_google_user(dict(claims))
    issued = _token_response(user)

    if config.OAUTH_SUCCESS_REDIRECT:
        separator = "&" if "?" in config.OAUTH_SUCCESS_REDIRECT else "?"
        return RedirectResponse(
            f"{config.OAUTH_SUCCESS_REDIRECT}{separator}token={issued.access_token}"
        )
    return issued


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


def _ingest_one_file(user_id: str, filename: str, content: bytes) -> UploadFileResult:
    """Store and index ``content`` as ``user_id``'s copy of ``filename``.

    Uses the structure-aware pipeline and REPLACES any previously indexed
    version of this user's copy of this filename -- re-uploading an edited
    document that produces fewer chunks would otherwise leave the surplus old
    chunks live and citable. Replacement is scoped to the owner, so it never
    touches another user's document of the same name.
    """
    result = documents.ingest_pdf_for_user(user_id, filename, content)
    return UploadFileResult(
        filename=filename,
        status="success",
        document_id=result["document_id"],
        pages_parsed=result["pages_parsed"],
        chunks_created=result["chunks_created"],
        chunks_indexed=result["chunks_indexed"],
        extraction_mode=result["extraction_mode"],
        table_cells=result.get("table_cells", 0),
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index one or more PDFs",
    description=(
        "Accepts one or more PDFs (multipart/form-data). Each is saved, parsed, "
        "chunked, embedded, and indexed AGAINST THE SIGNED-IN USER. Re-uploading "
        "the same filename replaces that user's own chunks only. One invalid file "
        "does not fail the rest of the batch -- check each entry's `status`."
    ),
)
async def upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more .pdf files."),
    user_id: str = Depends(get_current_user_id),
) -> UploadResponse:
    """Ingest, embed, and index each uploaded PDF; return a per-file summary.

    Colpali branch experiment: after a file's normal hybrid ingestion above
    succeeds -- unchanged, same result, same response shape -- its ColPali
    embedding + visual clustering is scheduled via ``BackgroundTasks`` (the
    same mechanism ``/ask`` already uses for summarization), so this response
    is never held up by ColPali's meaningfully heavier per-page model
    inference. A status row is written 'pending' synchronously, before this
    function returns, specifically so a client that immediately switches to
    RETRIEVAL_BACKEND=colpali and asks a question sees a clear
    still-indexing message (colpali_experiment.answer.STILL_INDEXING_MESSAGE)
    rather than a silent empty answer -- there is no gap where the status is
    simply absent. A missing colpali_experiment dependency (see
    ``_colpali_available`` above) skips this entirely; hybrid ingestion above
    is unaffected either way.
    """
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

            result = _ingest_one_file(user_id, safe_name, content)
            if _colpali_available and result.document_id:
                _colpali_set_ingest_status(
                    result.document_id, user_id, safe_name, status="pending"
                )
                background_tasks.add_task(
                    _colpali_ingest_for_upload, result.document_id, user_id, safe_name
                )
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

    with user_scope(user_id):
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
        f"Questions over {MAX_QUESTION_CHARS} characters are rejected. "
        "Retrieval covers ONLY the signed-in user's documents.\n\n"
        "Passing `session_id` turns this into a remembered turn: the question "
        "and answer are stored in that session, the session's running summary "
        "is used to resolve references (\"that policy\", \"the same band\"), and "
        "the summary is updated by a background task AFTER this response is "
        "sent -- summarization never adds to this call's latency.\n\n"
        "colpali branch experiment: `backend` selects which pipeline answers "
        "-- 'hybrid' (default, this description) or 'colpali' (a separate "
        "visual/page-image pipeline, colpali_experiment). Omit it to use "
        "RETRIEVAL_BACKEND's configured default. Session/memory persistence "
        "is identical either way."
    ),
    responses={404: {"model": ErrorResponse}},
)
def ask(
    payload: AskRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    backend: RetrievalBackend | None = Query(
        default=None,
        description=(
            "colpali branch experiment: override RETRIEVAL_BACKEND for this "
            "request only, e.g. for side-by-side testing without a restart."
        ),
    ),
) -> AskResponse | JSONResponse:
    """Answer ``payload.question``, optionally as a turn in ``payload.session_id``."""
    question = payload.question
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionTooLongError(
            f"Question is {len(question)} characters, which exceeds the "
            f"{MAX_QUESTION_CHARS}-character limit. Please shorten it."
        )

    # Per-request override takes precedence over the .env default -- the
    # brief's requirement that switching never needs a restart. Falls back to
    # "hybrid" (not whatever config.RETRIEVAL_BACKEND says) if the requested
    # backend is "colpali" but the experimental dependency isn't installed --
    # see _colpali_available above; the hybrid path must always work.
    active_backend: RetrievalBackend = backend or config.RETRIEVAL_BACKEND  # type: ignore[assignment]
    if active_backend == "colpali" and not _colpali_available:
        logger.warning(
            "RETRIEVAL_BACKEND=colpali requested but colpali_experiment is "
            "unavailable; falling back to hybrid."
        )
        active_backend = "hybrid"

    session = None
    if payload.session_id is not None:
        session = db.get_session(user_id, payload.session_id)
        if session is None:
            return JSONResponse(
                status_code=404, content=_error_body("not_found", "No such session.")
            )
        # Stored before generation runs: a session's memory must include the
        # user's own question even if generation then fails.
        db.add_message(user_id, session["id"], "user", question)

    # A question ABOUT the conversation ("what did I ask you earlier") has
    # nothing for the document pipeline to retrieve, so unhandled it would
    # reach query()'s fixed refusal -- misleading here, since the question
    # isn't unanswerable, it just isn't about the documents. Short-circuits
    # before user_scope/query() entirely; retrieval is never touched.
    if looks_like_meta_question(question):
        if session is not None:
            # Excludes the question just stored above -- otherwise the model
            # would be asked to recall a question from itself.
            transcript = (db.list_messages(user_id, session["id"]) or [])[:-1]
            answer_text = answer_meta_question(transcript, question)
            db.add_message(user_id, session["id"], "assistant", answer_text)
            background_tasks.add_task(
                summarize_turn_and_store,
                user_id,
                session["id"],
                session["summary"],
                question,
                answer_text,
            )
        else:
            # No session means no stored history at all -- honest about the
            # actual limitation, rather than the document-refusal sentence.
            answer_text = (
                "I don't have memory of earlier turns outside an active chat "
                "session. Start a chat session to ask about earlier turns."
            )
        return AskResponse(
            answer=answer_text,
            sources=[],
            query_type="conversation_meta",
            grounding=None,
            diagnostics=None,
            session_id=session["id"] if session else None,
        )

    if active_backend == "colpali":
        # colpali_experiment.answer.answer() does its own per-user filtering
        # at the store layer (store.get_user_embeddings(user_id) is scoped in
        # SQL), so it needs no user_scope() -- that context manager exists
        # specifically to filter Chroma/the hybrid stores, which this path
        # never touches.
        visual_answer = _colpali_answer(user_id, question)
        response = _colpali_to_rag_response(visual_answer)
    else:
        # The scope is bound around the whole pipeline rather than passed into
        # it: retrieval is final code and takes no user argument, and binding
        # here means every store read it makes -- dense, BM25, routing,
        # neighbour and parent expansion -- is filtered without any of those
        # modules being modified.
        with user_scope(user_id):
            # Blank questions raise InvalidQuestionError inside query() itself.
            response = query(
                question,
                conversation_context=session["summary"] if session else None,
                conversation_focus=session["last_document"] if session else None,
            )

    if session is not None:
        db.add_message(user_id, session["id"], "assistant", response.answer)
        if response.sources:
            # Synchronous and free of any LLM call, so it costs nothing on the
            # request's critical path; only the summary text needs the
            # background round-trip to an LLM. Identical regardless of which
            # backend produced `response` -- this reads only `.sources[0].source`.
            db.update_session_memory(
                user_id, session["id"], last_document=response.sources[0].source
            )
        # Scheduled via BackgroundTasks, which Starlette runs only AFTER the
        # response has been sent -- this is the NO-LAG gate. See
        # backend/memory.py's module docstring and
        # scripts/measure_memory_latency.py for the empirical check.
        background_tasks.add_task(
            summarize_turn_and_store,
            user_id,
            session["id"],
            session["summary"],
            question,
            response.answer,
        )

    return AskResponse(
        **response.to_dict(),
        session_id=session["id"] if session else None,
        backend=active_backend,
    )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List the signed-in user's documents",
    description=(
        "Reads the SQLite `documents` rows owned by the token's user and joins "
        "each to its live Chroma chunk count. Takes no user parameter."
    ),
)
def list_documents(user_id: str = Depends(get_current_user_id)) -> DocumentListResponse:
    rows = documents.list_documents_for_user(user_id)
    with user_scope(user_id):
        chunks = collection_stats()["count"]
        counts: dict[str, int] = {}
        for chunk in get_chunks_where(include=["metadatas"]):
            key = (chunk["metadata"] or {}).get("document_id")
            if key:
                counts[key] = counts.get(key, 0) + 1

    return DocumentListResponse(
        documents=[
            DocumentModel(
                id=row["id"],
                filename=row["filename"],
                created_at=row["created_at"],
                chunks=counts.get(row["id"], 0),
                size_bytes=row.get("size_bytes"),
            )
            for row in rows
        ],
        total_chunks=chunks,
        total_bytes=sum(row.get("size_bytes") or 0 for row in rows),
        documents_with_unknown_size=sum(
            1 for row in rows if row.get("size_bytes") is None
        ),
    )


@app.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and every chunk it produced",
    description=(
        "Removes the document's Chroma chunks and parent records first, then its "
        "SQLite row. Vectors go first deliberately: a crash between the two steps "
        "leaves a listed document with no vectors (unanswerable, and retryable), "
        "whereas the reverse order would strand retrievable, citable vectors with "
        "no row left to delete them by. A document belonging to another user "
        "returns 404 -- the response does not reveal that the id exists."
    ),
    responses={404: {"model": ErrorResponse}},
)
def delete_document(
    document_id: str, user_id: str = Depends(get_current_user_id)
) -> JSONResponse | DeleteDocumentResponse:
    result = documents.delete_document_for_user(user_id, document_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("not_found", "No such document."),
        )
    return DeleteDocumentResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunks_deleted=result["chunks_deleted"],
        row_deleted=True,
    )


# --------------------------------------------------------------------------
# Chat sessions and messages
#
# Memory is strictly per-session: a new session's `summary` starts '' and its
# `last_document` starts unset, so a fresh chat window has no memory of any
# other session. POST /ask (above) is where a session actually accrues memory;
# these endpoints create, list, and read what it accrued.
# --------------------------------------------------------------------------


@app.post("/sessions", response_model=SessionModel, status_code=201,
          summary="Start a chat session owned by the signed-in user")
def create_session(
    payload: SessionCreateRequest, user_id: str = Depends(get_current_user_id)
) -> SessionModel:
    return SessionModel(**db.create_session(user_id, payload.title))


@app.get(
    "/sessions",
    response_model=list[SessionModel],
    summary="The signed-in user's most recently active chat sessions",
    description=(
        "For the sidebar: the last `limit` sessions (default 10), most recently "
        "ACTIVE first -- a session's newest message time, falling back to its "
        "creation time if it has none yet. One indexed SQLite query; no vector "
        "store access."
    ),
)
def list_sessions(
    limit: int = 10, user_id: str = Depends(get_current_user_id)
) -> list[SessionModel]:
    return [SessionModel(**row) for row in db.list_sessions(user_id, limit=limit)]


@app.delete("/sessions/{session_id}", summary="Delete a session and its messages")
def delete_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    if not db.delete_session(user_id, session_id):
        return JSONResponse(
            status_code=404, content=_error_body("not_found", "No such session.")
        )
    # Messages go with it via ON DELETE CASCADE, not an application-level sweep.
    return {"session_id": session_id, "deleted": True}


@app.get(
    "/sessions/{session_id}/messages",
    response_model=list[MessageModel],
    summary="Read a session's messages",
    description=(
        "Messages have no owner column; ownership is resolved by joining to "
        "`sessions.user_id`. A session id belonging to someone else returns 404."
    ),
)
def get_messages(session_id: str, user_id: str = Depends(get_current_user_id)):
    rows = db.list_messages(user_id, session_id)
    if rows is None:
        return JSONResponse(
            status_code=404, content=_error_body("not_found", "No such session.")
        )
    return [MessageModel(**row) for row in rows]


@app.post(
    "/sessions/{session_id}/messages",
    response_model=MessageModel,
    status_code=201,
    summary="Append a message to a session the user owns",
)
def post_message(
    session_id: str,
    payload: MessageCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    row = db.add_message(user_id, session_id, payload.role, payload.content)
    if row is None:
        return JSONResponse(
            status_code=404, content=_error_body("not_found", "No such session.")
        )
    return MessageModel(**row)


# --------------------------------------------------------------------------
# Evaluation
#
# Read-only reporting over the evaluation harness in eval/eval_tool, plus the
# ability to start a run. Everything is scoped to the caller: a run asks
# questions against the caller's own documents, so listing someone else's
# numbers would be reporting on a corpus they cannot see.
# --------------------------------------------------------------------------


class EvalRunRequest(BaseModel):
    """Body for ``POST /eval/runs``."""

    test_set_id: str | None = Field(
        default=None,
        description=(
            "An uploaded test set (from POST /eval/test-sets). Omit to run the "
            "shipped gold set."
        ),
    )
    categories: list[str] | None = Field(
        default=None, description="Restrict the run to these categories."
    )
    skip_consistency_wait: bool = Field(
        default=True,
        description=(
            "Run the consistency pair back to back instead of waiting the "
            "configured ~5 minutes. Kept on by default so a browser-initiated "
            "run finishes in a reasonable time."
        ),
    )
    label: str = Field(default="", description="Optional name for this run.")


def _eval_error(exc: evaluation.EvalError) -> JSONResponse:
    return JSONResponse(status_code=400, content=_error_body("eval_error", str(exc)))


@app.get(
    "/eval/method",
    summary="How the evaluation scores answers, in plain English",
    description=(
        "Static explanation of the scoring process and metrics. Served from the "
        "backend so the wording cannot drift away from the code that implements "
        "it."
    ),
)
def eval_method(_user_id: str = Depends(get_current_user_id)) -> dict:
    return evaluation.METHOD_EXPLAINER


@app.get(
    "/eval/gold-set",
    summary="Category coverage of the shipped gold set",
)
def eval_gold_set(_user_id: str = Depends(get_current_user_id)) -> dict:
    return evaluation.gold_set_overview()


@app.get(
    "/eval/calibration",
    summary="The measured similarity distributions behind the threshold",
    description=(
        "Returns the correct-vs-wrong score distributions from the last "
        "calibration run, which is what justifies the pass threshold. 404 if "
        "calibration has never been run."
    ),
)
def eval_calibration(_user_id: str = Depends(get_current_user_id)):
    data = evaluation.calibration()
    if data is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("not_found", "No calibration has been run yet."),
        )
    # The per-pair detail is large and only the distributions are rendered.
    return {k: v for k, v in data.items() if k != "pairs"}


@app.get("/eval/runs", summary="List this user's evaluation runs, newest first")
def eval_runs(limit: int = 25, user_id: str = Depends(get_current_user_id)) -> dict:
    return {"runs": evaluation.list_runs(user_id, limit=max(1, min(limit, 100)))}


@app.get(
    "/eval/runs/latest",
    summary="The most recent evaluation run, in full",
)
def eval_latest(user_id: str = Depends(get_current_user_id)):
    run = evaluation.latest_run(user_id)
    if run is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("not_found", "No evaluation has been run yet."),
        )
    return run


@app.get(
    "/eval/comparison/latest",
    summary="Most recent ColPali-vs-Hybrid comparison report",
    description=(
        "colpali branch experiment: the most recent side-by-side comparison "
        "produced by `eval.eval_tool.run_comparison` from two saved eval "
        "runs (one per RETRIEVAL_BACKEND value) against the SAME gold set. "
        "Read-only, project-wide (not per-user) -- same auth as every other "
        "endpoint, no new unauthenticated route. 404 if no comparison has "
        "been generated yet."
    ),
)
def eval_comparison_latest(_user_id: str = Depends(get_current_user_id)):
    comparison = evaluation.latest_comparison()
    if comparison is None:
        return JSONResponse(
            status_code=404,
            content=_error_body(
                "not_found", "No ColPali-vs-Hybrid comparison has been generated yet."
            ),
        )
    return comparison


@app.get(
    "/eval/runs/{run_id}",
    summary="One evaluation run, including every per-question result",
)
def eval_run(run_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return evaluation.get_run(user_id, run_id)
    except evaluation.EvalError as exc:
        return JSONResponse(
            status_code=404, content=_error_body("not_found", str(exc))
        )


@app.post(
    "/eval/test-sets",
    status_code=201,
    summary="Upload a test set to evaluate against",
    description=(
        "Accepts a .json or .csv gold set. The file is validated against the "
        "schema before it is accepted, so a malformed set is rejected here with "
        "a message naming the problem rather than failing mid-run."
    ),
)
async def eval_upload_test_set(
    file: UploadFile = File(..., description="A .json or .csv test set."),
    user_id: str = Depends(get_current_user_id),
):
    try:
        content = await _read_limited(file, MAX_UPLOAD_BYTES)
        if not content:
            raise evaluation.EvalError("That file is empty.")
        return evaluation.save_test_set(user_id, file.filename or "", content)
    except evaluation.EvalError as exc:
        return _eval_error(exc)
    except UploadValidationError as exc:
        return _eval_error(evaluation.EvalError(str(exc)))
    finally:
        await file.close()


@app.post(
    "/eval/runs",
    status_code=202,
    summary="Start an evaluation run in the background",
    description=(
        "Returns immediately with a job id; poll GET /eval/jobs/{job_id} for "
        "progress. The run drives this same API over loopback using the "
        "caller's own token, so it measures exactly what this user experiences "
        "and sees only this user's documents."
    ),
)
def eval_start_run(
    payload: EvalRunRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    # The harness authenticates as this caller, so it needs the raw bearer
    # token. It is read back off the request rather than re-minted, so a run can
    # never outlive the credential that started it.
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        return _eval_error(evaluation.EvalError("Missing bearer token."))

    # The port is a launch-time choice, so the loopback base URL is taken from
    # the request the browser actually reached us on.
    base_url = f"http://127.0.0.1:{request.url.port or 8000}"

    try:
        job = evaluation.start_run(
            user_id,
            token,
            base_url,
            test_set_id=payload.test_set_id,
            categories=payload.categories,
            skip_consistency_wait=payload.skip_consistency_wait,
            label=payload.label,
        )
    except evaluation.EvalError as exc:
        return _eval_error(exc)
    return job.to_dict()


@app.get(
    "/eval/jobs/{job_id}",
    summary="Progress of a running evaluation",
)
def eval_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return evaluation.get_job(user_id, job_id).to_dict()
    except evaluation.EvalError as exc:
        return JSONResponse(
            status_code=404, content=_error_body("not_found", str(exc))
        )


@app.get(
    "/eval/jobs",
    summary="The caller's currently running evaluation, if any",
)
def eval_active_job(user_id: str = Depends(get_current_user_id)) -> dict:
    job = evaluation.active_job(user_id)
    return {"job": job.to_dict() if job else None}


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
