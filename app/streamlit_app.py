"""SmartDoc Streamlit UI.

A thin client: it holds no RAG logic and talks to the FastAPI backend over HTTP.

Run with:
    streamlit run app/streamlit_app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
HEALTH_TIMEOUT_SECONDS = 5

GLASS_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #141a35 45%, #1d1140 100%);
    color: #eef1ff;
}
.glass-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.glass-card h1, .glass-card h3 {
    margin: 0 0 0.35rem 0;
    color: #ffffff;
}
.glass-card p {
    margin: 0;
    color: rgba(238, 241, 255, 0.75);
}
.status-ok {
    color: #7ef0c0;
    font-weight: 600;
}
.status-bad {
    color: #ff9a9a;
    font-weight: 600;
}
</style>
"""


def check_backend(base_url: str) -> tuple[bool, str]:
    """Call the backend /health endpoint.

    Returns a (reachable, detail) pair so the caller can render either state.
    """
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/health",
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True, str(response.json())
    except requests.RequestException as exc:
        return False, str(exc)


def main() -> None:
    st.set_page_config(page_title="SmartDoc", page_icon="📄", layout="centered")
    st.markdown(GLASS_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h1>SmartDoc</h1>
            <p>Ask plain-English questions, get cited answers from your
            company documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reachable, detail = check_backend(BACKEND_URL)
    status_class = "status-ok" if reachable else "status-bad"
    status_text = "connected" if reachable else "unreachable"

    st.markdown(
        f"""
        <div class="glass-card">
            <h3>Backend status</h3>
            <p><code>{BACKEND_URL}</code> &middot;
            <span class="{status_class}">{status_text}</span></p>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not reachable:
        st.info("Start the API with: uvicorn backend.main:app --reload")


if __name__ == "__main__":
    main()
