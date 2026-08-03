"""Per-session conversational memory: a compact, continuously-updated summary.

The raw transcript is not what feeds the next turn -- ``sessions.summary`` is.
Storing the full history and re-sending it every turn would grow the prompt
without bound and dilute it with small talk; a single condensed paragraph,
rewritten each turn, stays a fixed, small cost regardless of session length.

NO-LAG GATE
-----------
This module's OpenAI call must never sit between the user and their answer.
:func:`summarize_turn_and_store` is the function scheduled via FastAPI
``BackgroundTasks`` from the ``/ask`` handler -- Starlette runs background tasks
only after the HTTP response has been sent, so the client's perceived latency
is exactly the retrieval-and-generation time from ``backend.rag.query``, with
this call adding nothing to it. See ``main.py``'s ``ask()`` for where it is
scheduled, and ``scripts/measure_memory_latency.py`` for the empirical check.

A background task's exceptions are logged, never raised into the ASGI
lifecycle: a failed summarization must not turn a successful answer into a
server error, and it must not corrupt the stored summary either -- the row is
only written once a valid replacement has been produced.
"""

from __future__ import annotations

import logging

import openai

import backend.config as config
from backend import db
from backend.vectorstore import _shared_openai

logger = logging.getLogger("smartdoc.memory")

# Bounds the summary's own growth: the prompt always asks for a summary within
# this budget, so re-summarising turn after turn cannot compound into an
# ever-larger block that eventually crowds out the retrieved context.
MAX_SUMMARY_CHARS = 800

_SUMMARY_PROMPT = f"""You maintain a running summary of one conversation with a \
company document assistant, used only to resolve references in later \
follow-up questions (pronouns, "that policy", "the same band").

Given the PREVIOUS summary and the LATEST question/answer turn, write an \
UPDATED summary that:
- Keeps only durable facts worth remembering for a follow-up: which \
document/topic/entity was discussed, and any specific value the user might \
refer back to (a band, a policy name, a figure).
- Drops small talk, phrasing, and anything not useful for resolving a future \
reference.
- Is a plain paragraph, at most {MAX_SUMMARY_CHARS} characters.
- If the turn added nothing worth remembering (e.g. it was refused, or purely \
conversational), return the previous summary unchanged.

Reply with the updated summary text only -- no labels, no quotes, no JSON."""


def summarize_turn(previous_summary: str, question: str, answer: str) -> str | None:
    """Produce the updated running summary for one turn, or None on failure.

    A single call that reads the old summary and writes a new one, rather than
    literally appending -- asking the model to re-condense every time is what
    keeps the result bounded (see ``MAX_SUMMARY_CHARS``) without separate
    truncation logic.
    """
    placeholder = "(none yet)"
    try:
        completion = _shared_openai().chat.completions.create(
            model=config.UTILITY_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Previous summary:\n{previous_summary or placeholder}\n\n"
                        f"Latest turn:\nQ: {question}\nA: {answer}"
                    ),
                },
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
    except openai.OpenAIError as exc:
        logger.warning("Session summarization call failed: %s", exc)
        return None
    if not text:
        return previous_summary
    # The model is told to echo the previous summary back verbatim when a turn
    # added nothing worth keeping. When there was no real previous summary, it
    # was shown `placeholder` instead of an empty string (a blank prompt field
    # reads as an omission, not "empty on purpose") -- so "unchanged" comes back
    # as the placeholder text itself. Un-echo it rather than storing prompt
    # scaffolding as if it were conversation content.
    if text == placeholder:
        return previous_summary
    return text[:MAX_SUMMARY_CHARS]


def summarize_turn_and_store(
    user_id: str, session_id: str, previous_summary: str, question: str, answer: str
) -> None:
    """Background-task entry point: summarize one turn and persist the result.

    Every failure mode is caught and logged here, never re-raised: this
    function runs after the response has already reached the client, so there
    is no request left to fail.
    """
    try:
        updated = summarize_turn(previous_summary, question, answer)
        if updated is None:
            return
        db.update_session_memory(user_id, session_id, summary=updated)
    except Exception:  # noqa: BLE001 -- a background task must never raise
        logger.exception(
            "Unhandled error summarizing session %s for user %s", session_id, user_id
        )
