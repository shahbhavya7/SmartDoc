"""HTTP client for the SmartDoc API.

This module is the entire backend integration. There is no retrieval, ranking, or
generation logic anywhere in streamlit_demo/ -- this app is a second, independent
client of the same FastAPI service the Next.js app talks to, which is the point:
if both work, the API's auth and isolation hold from more than the one client they
were built alongside.

Auth is a bearer token minted by scripts/mint_demo_token.py, read from .demo_token
on startup. A 401 is surfaced as TokenExpired so the UI can tell the user to
re-run that script, rather than crashing or silently falling back to another
identity.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_ROOT / ".demo_token"

API_BASE_URL = os.getenv("DEMO_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("DEMO_API_TIMEOUT_SECONDS", "180"))


class ApiError(Exception):
    """The API returned an error, or could not be reached."""


class TokenExpired(ApiError):
    """The demo token is missing, invalid, or expired. Re-run the mint script."""


class TokenMissing(TokenExpired):
    """No .demo_token file exists yet."""


def read_token() -> str:
    """The demo token from .demo_token.

    Raises TokenMissing rather than returning empty, so the caller cannot
    accidentally make unauthenticated calls that would 401 later with a less
    useful message.
    """
    if not TOKEN_FILE.exists():
        raise TokenMissing(
            "No demo token found. Run this first:\n\n"
            "    python scripts/mint_demo_token.py"
        )
    token = TOKEN_FILE.read_text().strip()
    if not token:
        raise TokenMissing(
            f"{TOKEN_FILE} is empty. Re-run: python scripts/mint_demo_token.py"
        )
    return token


def _request(
    method: str,
    path: str,
    token: str,
    *,
    json_body: dict | None = None,
    files: dict | None = None,
    timeout: int | None = None,
) -> object:
    """One authenticated call, with the API's error shape normalised."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            headers=headers,
            json=json_body,
            files=files,
            timeout=timeout or REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Could not reach the SmartDoc API at {API_BASE_URL}. "
            f"Is the backend running? ({exc})"
        ) from exc

    if response.status_code == 401:
        raise TokenExpired(
            "The demo token has expired or is invalid. Re-run:\n\n"
            "    python scripts/mint_demo_token.py"
        )

    if response.status_code == 204:
        return None

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if not response.ok:
        detail = ""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message", "")
        raise ApiError(detail or f"Request failed with status {response.status_code}.")

    return payload


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def whoami(token: str) -> dict:
    """The account this token belongs to. Used to confirm auth at startup."""
    return _request("GET", "/auth/me", token)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def list_documents(token: str) -> dict:
    return _request("GET", "/documents", token)


def upload_document(token: str, filename: str, content: bytes) -> dict:
    files = {"files": (filename, content, "application/pdf")}
    # Ingestion parses, chunks, and embeds, so it is far slower than a read.
    return _request("POST", "/upload", token, files=files, timeout=600)


def delete_document(token: str, document_id: str) -> dict:
    return _request("DELETE", f"/documents/{document_id}", token)


# ---------------------------------------------------------------------------
# Sessions and messages
# ---------------------------------------------------------------------------


def list_sessions(token: str, limit: int = 10) -> list[dict]:
    return _request("GET", f"/sessions?limit={limit}", token)


def create_session(token: str, title: str = "") -> dict:
    return _request("POST", "/sessions", token, json_body={"title": title})


def delete_session(token: str, session_id: str) -> dict:
    return _request("DELETE", f"/sessions/{session_id}", token)


def list_messages(token: str, session_id: str) -> list[dict]:
    return _request("GET", f"/sessions/{session_id}/messages", token)


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def ask(token: str, question: str, session_id: str | None = None) -> dict:
    """Ask a question. Passing session_id makes it a remembered turn."""
    return _request(
        "POST",
        "/ask",
        token,
        json_body={"question": question, "session_id": session_id},
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def list_eval_runs(token: str, limit: int = 25) -> dict:
    return _request("GET", f"/eval/runs?limit={limit}", token)


def get_eval_run(token: str, run_id: str) -> dict:
    return _request("GET", f"/eval/runs/{run_id}", token)


def start_eval_run(token: str, label: str = "") -> dict:
    return _request(
        "POST",
        "/eval/runs",
        token,
        json_body={
            "test_set_id": None,
            "categories": None,
            "label": label,
            "skip_consistency_wait": True,
        },
    )


def get_eval_job(token: str, job_id: str) -> dict:
    return _request("GET", f"/eval/jobs/{job_id}", token)
