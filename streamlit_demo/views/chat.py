"""Chat: sessions in the sidebar, answers with citations in the main area.

Citations are the point of this screen. Every generated answer renders its source
documents underneath it -- document name, page, section, and the snippet the
answer was grounded in -- inside an expander, so they are visually separate from
the answer text rather than mixed into it.

The citation data comes straight from the API response's `sources` array. Nothing
is inferred, matched, or reconstructed here: if the backend did not cite it, this
screen does not show it.
"""

from __future__ import annotations

import streamlit as st

from streamlit_demo import api

# The demo keeps a fixed-size window of sessions. Creating one beyond this
# deletes the least recently active, so the sidebar stays readable and the demo
# account does not accumulate history indefinitely.
#
# Enforced here rather than in the backend: this is a demo-client policy, and
# the real client deliberately has no such limit. It uses the ordinary DELETE
# endpoint, so the message cascade is the server's, not a second implementation.
MAX_SESSIONS = 5


def _load_sessions(token: str) -> list[dict]:
    try:
        return api.list_sessions(token, limit=MAX_SESSIONS)
    except api.ApiError as exc:
        st.sidebar.error(str(exc))
        return []


def _prune_to_limit(token: str, keep: int = MAX_SESSIONS) -> int:
    """Delete the oldest sessions beyond ``keep``. Returns how many went.

    list_sessions orders most-recently-active first, so anything past the limit
    is the least recently used. A large limit is requested here on purpose --
    asking for only MAX_SESSIONS would never reveal the extras that need
    deleting.
    """
    try:
        existing = api.list_sessions(token, limit=100)
    except api.ApiError:
        return 0

    removed = 0
    for session in existing[keep:]:
        try:
            api.delete_session(token, session["id"])
            removed += 1
        except api.ApiError:
            # One failed delete should not block the rest, nor the new session
            # the caller is about to create.
            continue
    return removed


def _session_label(session: dict) -> str:
    title = (session.get("title") or "").strip()
    return title if title else "New chat"


def render_citations(sources: list[dict]) -> None:
    """Render the citation block under an answer.

    Kept in an expander and labelled with the count so the answer stays readable
    while the evidence is always one click away, never hidden and never merged
    into the prose.
    """
    if not sources:
        st.caption("No sources cited. The answer was not grounded in a document.")
        return

    with st.expander(f"Sources ({len(sources)})", expanded=True):
        for index, source in enumerate(sources, start=1):
            page = source.get("page")
            page_end = source.get("page_end")
            if page_end and page_end != page:
                pages = f"pages {page}-{page_end}"
            else:
                pages = f"page {page}"

            section = (source.get("section") or "").strip()
            heading = f"{index}. {source.get('source', 'unknown')} | {pages}"
            if section:
                heading += f" | {section}"

            st.markdown(f"**{heading}**")
            snippet = (source.get("snippet") or "").strip()
            if snippet:
                # Quoted rather than plain text so the excerpt reads as evidence
                # rather than as more of the answer.
                st.caption(snippet)
            if index < len(sources):
                st.divider()


def _render_turn(message: dict) -> None:
    role = "user" if message.get("role") == "user" else "assistant"
    with st.chat_message(role):
        # The backend already decided the format -- tables, lists, prose -- so
        # this renders the markdown it returned and adds no formatting of its own.
        st.markdown(message.get("content", ""))
        if role == "assistant" and message.get("sources") is not None:
            render_citations(message.get("sources") or [])


def render(token: str) -> None:
    st.subheader("Chat")

    # -- Session list -----------------------------------------------------
    st.sidebar.subheader("Sessions")
    if st.sidebar.button("New chat", use_container_width=True):
        try:
            # Prune BEFORE creating, keeping room for the new one, so the total
            # lands at exactly MAX_SESSIONS rather than briefly exceeding it.
            dropped = _prune_to_limit(token, keep=MAX_SESSIONS - 1)
            created = api.create_session(token)
            st.session_state["session_id"] = created["id"]
            st.session_state["history"] = []
            if dropped:
                st.session_state["pruned_notice"] = dropped
            st.rerun()
        except api.ApiError as exc:
            st.sidebar.error(str(exc))

    dropped = st.session_state.pop("pruned_notice", 0)
    if dropped:
        st.sidebar.caption(
            f"Removed {dropped} old session(s) to stay within the "
            f"{MAX_SESSIONS}-session limit."
        )

    sessions = _load_sessions(token)
    current = st.session_state.get("session_id")

    if sessions:
        ids = [s["id"] for s in sessions]
        labels = {s["id"]: _session_label(s) for s in sessions}
        index = ids.index(current) if current in ids else 0
        chosen = st.sidebar.radio(
            "Recent sessions",
            ids,
            index=index,
            format_func=lambda sid: labels.get(sid, sid[:8]),
            label_visibility="collapsed",
        )
        if chosen != current:
            # Switching sessions loads that session's own history from the
            # server, which is what makes each session's memory independent.
            st.session_state["session_id"] = chosen
            st.session_state.pop("history", None)
            st.rerun()
        current = st.session_state.get("session_id") or chosen
    else:
        st.sidebar.caption("No sessions yet. Use New chat to start one.")

    if not current:
        st.info("Select a session in the sidebar, or start a new chat.")
        return

    # -- History ----------------------------------------------------------
    if "history" not in st.session_state:
        try:
            stored = api.list_messages(token, current)
        except api.ApiError as exc:
            st.error(str(exc))
            stored = []
        # Stored messages carry no citations; only answers produced in this
        # session render a source block. Marking them None (not []) keeps
        # "not recorded" distinct from "cited nothing".
        st.session_state["history"] = [
            {"role": m["role"], "content": m["content"], "sources": None}
            for m in stored
        ]

    for message in st.session_state["history"]:
        _render_turn(message)

    # -- Ask --------------------------------------------------------------
    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state["history"].append(
        {"role": "user", "content": question, "sources": None}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating"):
            try:
                response = api.ask(token, question, session_id=current)
            except api.ApiError as exc:
                st.error(str(exc))
                return

        answer = response.get("answer", "")
        sources = response.get("sources") or []
        st.markdown(answer)
        render_citations(sources)

    st.session_state["history"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
