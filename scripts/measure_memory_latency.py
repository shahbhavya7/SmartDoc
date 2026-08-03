"""Empirically confirm the NO-LAG gate: session memory adds no perceived latency.

``BackgroundTasks`` (used by ``POST /ask`` for summarization) is documented to
run only after the response has been sent -- but that guarantee is invisible to
an in-process TestClient, which awaits the whole ASGI call, background tasks
included, before returning. Proving it for real needs a real server and a real
socket: this script fires requests at a running ``uvicorn`` process over HTTP
and times them from outside the process, so the client-perceived latency is
whatever actually crossed the wire.

Compares:
    A. stateless /ask (no session_id -- Phase 1 behaviour, no memory at all)
    B. first turn in a session (stores messages, schedules summarization)
    C. second turn in the SAME session (also reads back the stored summary)

If summarization were awaited inline instead of scheduled as a background
task, B and C would each be slower than A by roughly one extra UTILITY_MODEL
call (typically several hundred ms). This script reports the medians and their
deltas so that regression is visible as a number, not an assumption.

Usage:
    .venv/bin/python -m scripts.measure_memory_latency \\
        --email dev@smartdoc.local --password devpassword123 \\
        --question "How many days of annual leave do Standard band employees get?"
"""

from __future__ import annotations

import argparse
import statistics
import time

import requests

import backend.config as config


def _login(base_url: str, email: str, password: str) -> str:
    response = requests.post(
        f"{base_url}/auth/login", json={"email": email, "password": password}, timeout=30
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _timed_ask(base_url: str, token: str, question: str, session_id: str | None) -> float:
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/ask",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config.BACKEND_URL)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--question", default="How many days of annual leave do Standard band employees get?"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    token = _login(args.base_url, args.email, args.password)

    stateless = [
        _timed_ask(args.base_url, token, args.question, None) for _ in range(args.repeats)
    ]

    session = requests.post(
        f"{args.base_url}/sessions",
        json={"title": "latency probe"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    ).json()

    first_turn = [_timed_ask(args.base_url, token, args.question, session["id"])]
    # Give the background summarization a moment to land before the next turn
    # reads it back, so turn C reflects steady-state, memory-bearing behaviour.
    time.sleep(2)
    later_turns = [
        _timed_ask(args.base_url, token, args.question, session["id"])
        for _ in range(args.repeats)
    ]

    def _fmt(label: str, values: list[float]) -> None:
        print(
            f"  {label:32} median={statistics.median(values) * 1000:6.0f}ms  "
            f"n={len(values)}  values={[round(v * 1000) for v in values]}"
        )

    print(f"\nQuestion: {args.question!r}\n")
    _fmt("A. stateless (no session)", stateless)
    _fmt("B. session, 1st turn", first_turn)
    _fmt("C. session, later turns", later_turns)

    base_median = statistics.median(stateless)
    later_median = statistics.median(later_turns)
    delta_ms = (later_median - base_median) * 1000
    print(
        f"\nDelta (C - A): {delta_ms:+.0f}ms. If session memory added summarization "
        "to the request path, this would be roughly one extra LLM call slower "
        "(typically several hundred ms), not noise-level."
    )


if __name__ == "__main__":
    main()
