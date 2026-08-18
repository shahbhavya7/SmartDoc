"""SmartDoc -- Streamlit demo client.

A deliberately basic second client for the SmartDoc API. It exists to demonstrate
the deliverable "Streamlit UI with source citation" and to prove the backend works
from a client other than the Next.js app it was built alongside.

Everything here is HTTP. There is no retrieval, ranking, chunking, or generation
code in this folder -- the backend decides all of that, including how an answer is
formatted, and this app renders what comes back.

Authentication is a token minted locally by scripts/mint_demo_token.py for a real
pre-existing account. There is no login form by design: the demo is a single
pre-selected user.

Run with:
    streamlit run streamlit_demo/app.py

This is independent of ./run.sh (the Next.js stack). Both can run at once against
the same backend, because both are ordinary API clients.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set BEFORE anything imports pyarrow (streamlit does, to render dataframes).
#
# PyArrow 25 defaults to the mimalloc allocator, which segfaults on macOS when
# it initialises on one of Streamlit's worker threads -- the crash is inside
# mi_thread_init, reached from NumPyConverter::Convert while turning a pandas
# frame into an Arrow table. It takes the whole process down, so the app dies
# mid-session with "Python quit unexpectedly".
#
# The system allocator has no such thread-init path. Arrow is only used here to
# render two small tables, so the allocator choice costs nothing.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Optional: .env is convenience, not a requirement.
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from streamlit_demo import api  # noqa: E402
from streamlit_demo.views import chat, documents, evaluation  # noqa: E402

st.set_page_config(page_title="SmartDoc Demo", layout="wide")


def authenticate() -> tuple[str, dict] | None:
    """Sign in as the demo user, silently, and cache the result.

    Returns (token, user) or None when the token is missing/expired, in which
    case the instructions are already on screen. Nothing here prompts for
    credentials -- the demo is one pre-configured account.
    """
    if "token" in st.session_state and "user" in st.session_state:
        return st.session_state["token"], st.session_state["user"]

    try:
        token = api.read_token()
        user = api.whoami(token)
    except api.TokenExpired as exc:
        st.error(str(exc))
        st.caption(
            "The demo account signed up with Google and has no password, so the "
            "token is issued locally rather than through /auth/login."
        )
        return None
    except api.ApiError as exc:
        st.error(str(exc))
        return None

    st.session_state["token"] = token
    st.session_state["user"] = user
    return token, st.session_state["user"]


def sidebar_info(token: str, user: dict) -> None:
    """Signed-in account plus document and session counts. Plain metrics."""
    st.sidebar.header("SmartDoc Demo")
    st.sidebar.text(user.get("email", "unknown"))

    doc_count = session_count = None
    try:
        listing = api.list_documents(token)
        doc_count = len(listing.get("documents", []))
        st.session_state["total_chunks"] = listing.get("total_chunks", 0)
    except api.ApiError:
        pass
    try:
        session_count = len(api.list_sessions(token, limit=100))
    except api.ApiError:
        pass

    left, right = st.sidebar.columns(2)
    left.metric("Documents", doc_count if doc_count is not None else "-")
    right.metric("Sessions", session_count if session_count is not None else "-")
    st.sidebar.caption(
        f"{st.session_state.get('total_chunks', 0)} indexed chunks | "
        f"API {api.API_BASE_URL}"
    )
    st.sidebar.divider()


def main() -> None:
    credentials = authenticate()
    if credentials is None:
        st.stop()
    token, user = credentials

    sidebar_info(token, user)

    # Sidebar radio rather than st.tabs: st.tabs always opens on its first tab
    # and offers no way to choose which, so a reload could not land on
    # Documents. A radio holds the choice in session state, which also means
    # only the selected view's API calls run -- st.tabs renders every tab's
    # body on every rerun, so the evaluation report was being fetched even
    # while reading the chat.
    page = st.sidebar.radio(
        "View",
        ("Documents", "Chat", "Evaluation"),
        index=0,
        key="page",
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    if page == "Documents":
        documents.render(token)
    elif page == "Chat":
        chat.render(token)
    else:
        evaluation.render(token)


# Called unconditionally, with no __main__ guard. `streamlit run` executes this
# file with __name__ == "main" (verified -- not "__main__"), so the usual guard
# is simply False here and the app would never start behind it.
main()
