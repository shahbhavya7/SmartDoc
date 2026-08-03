"""Interactive demo of per-session chat memory (V2 Phase 2, Part A).

A small CLI against the live API, not a UI: the Streamlit client predates
authentication (see DECISIONS.md's Phase 1 "known breakage, accepted") and is
replaced by the Next.js client in a later V2 phase, so building this into
``app/streamlit_app.py`` would be work outside this phase's scope. This script
demonstrates the feature the same way `scripts/measure_memory_latency.py`
measures it: as a real HTTP client of the running FastAPI server.

What it shows
--------------
1. A fresh session starts with no memory of any other session.
2. A follow-up question in the SAME session resolves a bare reference
   ("And the Executive band?") using the running summary -- printed after each
   turn, so you can see it update.
3. The summarization that produced that update ran in the background: this
   script's own wall-clock time per turn is displayed, and it does not include
   a visible pause for the summary call.

Usage:
    .venv/bin/uvicorn backend.main:app --reload   # in one terminal
    .venv/bin/python -m scripts.demo_chat_memory   # in another

    # bring your own questions:
    .venv/bin/python -m scripts.demo_chat_memory \\
        --question "How many sick days do I get?" \\
        --question "And what if I'm a contractor?"

    # or drop into a REPL against a fresh session:
    .venv/bin/python -m scripts.demo_chat_memory --interactive
"""

from __future__ import annotations

import argparse
import time

import requests

import backend.config as config

DEFAULT_QUESTIONS = [
    "How many days of annual leave do Standard band employees get?",
    # A bare follow-up: only resolvable via the session's running summary,
    # since neither "annual leave" nor "Standard band" is repeated here.
    "And what about the Executive band?",
]


def _login_or_signup(base_url: str, email: str, password: str) -> str:
    response = requests.post(
        f"{base_url}/auth/login", json={"email": email, "password": password}, timeout=30
    )
    if response.status_code == 401:
        response = requests.post(
            f"{base_url}/auth/signup",
            json={"email": email, "password": password},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()["access_token"]


def _ask(base_url: str, token: str, question: str, session_id: str) -> dict:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/ask",
        json={"question": question, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


def _session_summary(base_url: str, token: str, session_id: str) -> str:
    response = requests.get(
        f"{base_url}/sessions", headers={"Authorization": f"Bearer {token}"}, timeout=30
    )
    response.raise_for_status()
    for session in response.json():
        if session["id"] == session_id:
            return session["summary"]
    return ""


def run_turn(base_url: str, token: str, session_id: str, question: str) -> None:
    print(f"\n> {question}")
    body, elapsed_ms = _ask(base_url, token, question, session_id)
    print(f"{body['answer']}")
    if body["sources"]:
        cited = ", ".join(f"{s['source']} p{s['page']}" for s in body["sources"][:3])
        print(f"  (cited: {cited})")
    print(f"  [{elapsed_ms:.0f}ms]")

    # Give the background summarization a moment to land before printing it --
    # purely so the demo's OWN print statement has something to show; the /ask
    # call above already returned without waiting for this.
    time.sleep(1.5)
    summary = _session_summary(base_url, token, session_id)
    print(f"  session memory now: {summary!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config.BACKEND_URL)
    parser.add_argument("--email", default="demo@smartdoc.local")
    parser.add_argument("--password", default="demo-password-123")
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Add a question to the scripted turn sequence (repeatable).",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Read questions from stdin instead."
    )
    args = parser.parse_args()

    token = _login_or_signup(args.base_url, args.email, args.password)
    session = requests.post(
        f"{args.base_url}/sessions",
        json={"title": "memory demo"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    ).json()
    print(f"New session {session['id']} (starts with no memory of any other session)")

    if args.interactive:
        print("Type a question and press Enter (Ctrl-D to quit).")
        try:
            while True:
                question = input("\n> ").strip()
                if question:
                    run_turn(args.base_url, token, session["id"], question)
        except EOFError:
            pass
    else:
        for question in args.questions or DEFAULT_QUESTIONS:
            run_turn(args.base_url, token, session["id"], question)


if __name__ == "__main__":
    main()
