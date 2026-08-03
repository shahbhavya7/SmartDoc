"""SmartDoc Streamlit UI.

A thin client: it holds NO retrieval or generation logic of its own. Every
document upload and every question is sent to the FastAPI backend over HTTP,
and this module only renders what comes back. It must stay operable by a
non-technical user with zero explanation, so failures are always shown as a
plain-English message -- never a traceback or a blank screen.

Layout:
    Sidebar -- upload PDFs (POST /upload) and see backend connection status.
    Main area -- a chat conversation (POST /ask) with a citations block under
                 every answer.

Run with:
    streamlit run app/streamlit_app.py

Configuration:
    SMARTDOC_API_URL -- base URL of the FastAPI backend. Falls back to the
                         legacy BACKEND_URL variable, then to the local
                         default, so existing .env files keep working.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv(
    "SMARTDOC_API_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
).rstrip("/")

# /ask can involve several sequential OpenAI calls (classification, retrieval
# rerank, generation, grounding check) so the client timeout is generous and
# deliberately looser than any single backend call's own timeout.
HEALTH_TIMEOUT_SECONDS = 5
ASK_TIMEOUT_SECONDS = 180
UPLOAD_TIMEOUT_SECONDS = 300


# --------------------------------------------------------------------------
# Backend calls -- the only place this app talks to the network.
# --------------------------------------------------------------------------


def _extract_error_message(response: requests.Response) -> str:
    """Pull a friendly message out of the backend's ``ErrorResponse`` shape.

    Falls back to a generic, non-technical message when the body is not JSON
    or does not match the expected ``{"error": {"type", "message"}}`` shape.
    """
    try:
        body = response.json()
        message = body.get("error", {}).get("message")
        if message:
            return message
    except ValueError:
        pass
    return (
        f"The backend reported an error (HTTP {response.status_code}). "
        "Please try again in a moment."
    )


def check_backend_health(base_url: str) -> tuple[bool, str]:
    """Call ``GET /health``. Returns (reachable, detail) -- never raises."""
    try:
        response = requests.get(f"{base_url}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return True, str(response.json())
    except requests.RequestException as exc:
        return False, str(exc)


def call_ask(question: str, base_url: str) -> dict[str, Any]:
    """POST ``question`` to ``/ask``.

    Returns ``{"ok": True, "data": <AskResponse dict>}`` on success, or
    ``{"ok": False, "message": <friendly string>}`` on any failure. Never
    raises -- every network, timeout, HTTP, and JSON-decoding failure is
    caught here so the UI can always render something sensible.
    """
    try:
        response = requests.post(
            f"{base_url}/ask", json={"question": question}, timeout=ASK_TIMEOUT_SECONDS
        )
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "message": (
                "Could not reach the SmartDoc backend. Please check that it "
                "is running and try again."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "message": (
                "The backend took too long to respond. It may be busy -- "
                "please try again."
            ),
        }
    except requests.exceptions.RequestException:
        return {
            "ok": False,
            "message": "Could not reach the backend. Please try again.",
        }

    if response.status_code == 200:
        try:
            return {"ok": True, "data": response.json()}
        except ValueError:
            return {
                "ok": False,
                "message": "The backend sent back a response we could not read.",
            }
    return {"ok": False, "message": _extract_error_message(response)}


def call_upload(files: list[Any], base_url: str) -> dict[str, Any]:
    """POST one or more uploaded PDFs to ``/upload``.

    ``files`` is a list of Streamlit ``UploadedFile`` objects. Returns the
    same ``{"ok": ..., ...}`` shape as :func:`call_ask`.
    """
    multipart = [
        ("files", (f.name, f.getvalue(), "application/pdf")) for f in files
    ]
    try:
        response = requests.post(
            f"{base_url}/upload", files=multipart, timeout=UPLOAD_TIMEOUT_SECONDS
        )
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "message": (
                "Could not reach the SmartDoc backend. Please check that it "
                "is running and try again."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "message": (
                "The upload took too long. Try uploading fewer or smaller "
                "files at a time."
            ),
        }
    except requests.exceptions.RequestException:
        return {
            "ok": False,
            "message": "Could not reach the backend. Please try again.",
        }

    if response.status_code == 200:
        try:
            return {"ok": True, "data": response.json()}
        except ValueError:
            return {
                "ok": False,
                "message": "The backend sent back a response we could not read.",
            }
    return {"ok": False, "message": _extract_error_message(response)}


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render the citations block under an answer, visually distinct from it."""
    if not sources:
        st.caption(
            "No sources -- this answer could not be grounded in the uploaded "
            "documents."
        )
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, start=1):
            page = source.get("page")
            page_end = source.get("page_end")
            page_label = f"page {page}" if page is not None else "page unknown"
            if page_end and page_end != page:
                page_label = f"pages {page}-{page_end}"

            st.markdown(f"**{i}. {source.get('source', 'unknown document')} -- {page_label}**")
            if source.get("section"):
                st.caption(source["section"])
            snippet = source.get("snippet", "")
            if snippet:
                st.caption(f"“{snippet}”")
            if i < len(sources):
                st.divider()


def render_ask_result(result: dict[str, Any]) -> None:
    """Render one assistant turn: the answer, then its citations block."""
    if not result.get("ok"):
        st.error(result.get("message", "Something went wrong. Please try again."))
        return

    data = result["data"]
    st.write(data.get("answer", ""))
    render_sources(data.get("sources") or [])

    grounding = data.get("grounding") or {}
    if grounding.get("note"):
        st.caption(f"Note: {grounding['note']}")


def render_upload_result(result: dict[str, Any]) -> None:
    """Render per-file upload outcomes -- some files can succeed, some fail."""
    if not result.get("ok"):
        st.error(result.get("message", "Upload failed. Please try again."))
        return

    data = result["data"]
    for file_result in data.get("files", []):
        name = file_result.get("filename", "unknown file")
        if file_result.get("status") == "success":
            chunks = file_result.get("chunks_indexed", 0)
            st.success(f"'{name}' uploaded and indexed ({chunks} chunks).")
        else:
            st.error(f"'{name}' failed: {file_result.get('error', 'unknown error')}")

    st.caption(
        f"Document library now has {data.get('collection_count', '?')} indexed "
        "chunks."
    )


# --------------------------------------------------------------------------
# Styling -- "liquid glass" theme. CSS only: no functional/logic changes.
# Targets Streamlit's stable container classes / data-testid attributes so
# the effect survives Streamlit's own DOM without touching any Python
# rendering logic above.
# --------------------------------------------------------------------------

LIQUID_GLASS_CSS = """
<style>
/* Dark gradient backdrop behind the whole app */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, #24314f 0%, #141a2e 45%, #0a0d18 100%);
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Faint ruled-paper motif behind the main content -- restrained accent */
[data-testid="stAppViewContainer"] > .main {
    background-image: repeating-linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.025) 0px,
        rgba(255, 255, 255, 0.025) 1px,
        transparent 1px,
        transparent 34px
    );
    background-position: 0 90px;
}

/* Base text legibility over the dark gradient */
[data-testid="stAppViewContainer"] * {
    color: #eef1f8;
}

/* Sidebar: frosted glass panel */
[data-testid="stSidebar"] {
    background: rgba(24, 28, 46, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Chat bubbles: frosted glass cards */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}

/* Citation panel (expander) -- visually distinct glass card */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.045);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(200, 180, 140, 0.28);
    border-left: 3px solid rgba(200, 180, 140, 0.55);
    border-radius: 10px;
    box-shadow: 0 2px 16px rgba(0, 0, 0, 0.3);
}

[data-testid="stExpander"] summary {
    color: #f4e9d0;
}

/* Chat input: frosted glass bar */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 14px;
}

[data-testid="stChatInput"] textarea {
    color: #eef1f8 !important;
}

/* Alerts (error / success / info) must stay clearly legible -- solid,
   high-contrast backgrounds rather than translucent glass. */
[data-testid="stAlert"] {
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.18);
}

div[data-testid="stAlertContainer"][data-baseweb="notification"] {
    color: #0d1117;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #0d1117 !important;
    font-weight: 500;
}

/* Buttons -- subtle glass with a clear hover state */
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: #eef1f8;
}

/* File uploader drop zone -- glass card */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px dashed rgba(255, 255, 255, 0.25);
    border-radius: 12px;
}

/* Title and captions -- ensure strong contrast */
h1, h2, h3 {
    color: #ffffff;
}
</style>
"""


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def render_sidebar() -> None:
    """Upload panel and backend connection status."""
    with st.sidebar:
        st.header("1. Add your documents")
        st.caption(
            "Upload PDF files -- HR policies, manuals, onboarding guides, "
            "SOPs -- so SmartDoc can answer questions about them."
        )
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("Upload", disabled=not uploaded_files, use_container_width=True):
            with st.spinner("Uploading and indexing document(s)..."):
                result = call_upload(uploaded_files, API_BASE_URL)
            render_upload_result(result)

        st.divider()
        st.caption("Connection")
        reachable, detail = check_backend_health(API_BASE_URL)
        if reachable:
            st.success("Backend connected")
        else:
            st.error(
                "Backend not reachable. Please make sure the SmartDoc "
                "service is running, then refresh this page."
            )
            st.caption(detail)


def main() -> None:
    st.set_page_config(page_title="SmartDoc", page_icon="\U0001F4C4", layout="centered")
    st.markdown(LIQUID_GLASS_CSS, unsafe_allow_html=True)

    st.title("SmartDoc")
    st.caption(
        "Ask plain-English questions about your company documents and get "
        "answers with sources."
    )

    render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.info(
            "2. Type a question below, e.g. “How many days of annual "
            "leave do I get?”"
        )

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_ask_result(turn["result"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_ask(question, API_BASE_URL)
            render_ask_result(result)
        st.session_state.history.append({"question": question, "result": result})


if __name__ == "__main__":
    main()
