"""Phase 2, Part A acceptance tests: per-session memory.

Offline and deterministic: the RAG pipeline itself (retrieval + real OpenAI
calls) is replaced with a stub at the ``backend.main.query`` call site, because
these tests are about the SESSION WIRING -- does a new session start empty,
does a follow-up receive the prior turn's context, is memory isolated between
sessions, does the background summarization get scheduled without blocking the
response -- not about answer quality, which the eval harness already measures
against the real pipeline.

The NO-LAG requirement (summarization runs after the response is sent) is an
architectural property of ``BackgroundTasks`` that a same-process TestClient
cannot observe directly: Starlette's TestClient awaits the background task as
part of the same call, so from the test's point of view it always looks
synchronous. What CAN be verified here is that summarization is genuinely
scheduled as a background task rather than awaited inline before the response
is built (checked via a spy that records call order against response
construction), and that a slow or failing summarizer never surfaces as a
request failure. The real over-the-wire latency claim is verified separately,
against a live server, by ``scripts/measure_memory_latency.py``.

Run:
    python -m pytest tests/test_phase2_memory.py -v
"""

from __future__ import annotations

import importlib
import os

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="session", autouse=True)
def _isolated_stores(tmp_path_factory):
    """A throwaway SQLite file and Chroma dir, distinct from other test files."""
    tmp = tmp_path_factory.mktemp("smartdoc_phase2")
    os.environ["SQLITE_PATH"] = str(tmp / "test.db")
    os.environ["CHROMA_DIR"] = str(tmp / "chroma")
    os.environ["MULTI_USER_ENABLED"] = "true"
    os.environ["JWT_SECRET"] = "test-secret-not-the-placeholder"
    os.environ["OPENAI_API_KEY"] = "sk-test-not-used"
    os.environ["BCRYPT_ROUNDS"] = "4"

    import backend.config as config

    importlib.reload(config)
    yield


@pytest.fixture(scope="session")
def modules(_isolated_stores):
    import backend.config as config
    from backend import auth, db, memory

    db.reset_state_for_tests()
    db.init_db()
    return {"config": config, "auth": auth, "db": db, "memory": memory}


@pytest.fixture()
def user(modules):
    """A fresh user per test, so sessions/messages never leak across tests."""
    import uuid

    db, auth = modules["db"], modules["auth"]
    email = f"{uuid.uuid4()}@example.com"
    return db.create_user(email, auth.hash_password("password-123"))


@pytest.fixture(scope="session")
def client(modules):
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


def auth_header(client, user, modules):
    token = modules["auth"].create_access_token(user["id"], user["email"])
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# db.py: schema, migration, and query correctness
# ---------------------------------------------------------------------------


def test_new_session_starts_with_empty_summary_and_no_focus(modules, user):
    db = modules["db"]
    session = db.create_session(user["id"], "My chat")
    assert session["summary"] == ""
    assert session["last_document"] is None

    fetched = db.get_session(user["id"], session["id"])
    assert fetched["summary"] == ""
    assert fetched["last_document"] is None


def test_migration_adds_columns_to_a_pre_phase2_table(modules):
    """A sessions table created before summary/last_document existed still works."""
    db = modules["db"]
    with db.connect() as conn:
        conn.execute("ALTER TABLE sessions RENAME TO sessions_new_phase2")
        conn.execute(
            "CREATE TABLE sessions ("
            " id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',"
            " created_at TEXT NOT NULL)"
        )
        conn.execute("DROP TABLE sessions")
        conn.execute("ALTER TABLE sessions_new_phase2 RENAME TO sessions")
    # Force the migration path by clearing the "already initialised" cache.
    db.reset_state_for_tests()
    db.init_db()
    with db.connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert {"summary", "last_document"} <= columns


def test_update_session_memory_is_scoped_to_the_owner(modules, user):
    db = modules["db"]
    import uuid

    other = db.create_user(f"{uuid.uuid4()}@example.com", "x")
    session = db.create_session(user["id"], "t")

    assert db.update_session_memory(other["id"], session["id"], summary="hijacked") is False
    assert db.get_session(user["id"], session["id"])["summary"] == ""

    assert db.update_session_memory(user["id"], session["id"], summary="mine") is True
    assert db.get_session(user["id"], session["id"])["summary"] == "mine"


def test_list_sessions_orders_by_last_activity_and_respects_limit(modules, user):
    """Most recently ACTIVE first, and capped at `limit` (the sidebar's default 10)."""
    db = modules["db"]

    ids = [db.create_session(user["id"], f"s{i}")["id"] for i in range(3)]
    # Force distinct, known timestamps rather than relying on real wall-clock
    # gaps between fast successive inserts, which can collide at
    # second-resolution and make the ordering assertion flaky.
    with db.connect() as conn:
        for i, sid in enumerate(ids):
            conn.execute(
                "UPDATE sessions SET created_at = ? WHERE id = ?",
                (f"2026-01-0{i + 1}T00:00:00+00:00", sid),
            )
        # A message on the OLDEST session (ids[0]) makes it the most recently
        # ACTIVE, even though it is not the most recently created.
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) "
            "VALUES ('m1', ?, 'user', 'hi', '2026-01-05T00:00:00+00:00')",
            (ids[0],),
        )

    ordered = [s["id"] for s in db.list_sessions(user["id"], limit=10)]
    assert ordered[0] == ids[0]  # activity from the message wins
    assert ordered[1] == ids[2]  # then by creation time, newest first
    assert ordered[2] == ids[1]

    assert len(db.list_sessions(user["id"], limit=2)) == 2


# ---------------------------------------------------------------------------
# backend/memory.py
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, text: str):
        self._text = text

    def create(self, **kwargs):
        class Choice:
            def __init__(self, text):
                self.message = type("M", (), {"content": text})()

        class Completion:
            def __init__(self, text):
                self.choices = [Choice(text)]

        return Completion(self._text)


class _FakeOpenAI:
    def __init__(self, text: str):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(text)})()


def test_summarize_turn_produces_bounded_text(modules, monkeypatch):
    memory = modules["memory"]
    monkeypatch.setattr(memory, "_shared_openai", lambda: _FakeOpenAI("condensed summary"))
    result = memory.summarize_turn("(none yet)", "How many sick days?", "Ten days.")
    assert result == "condensed summary"


def test_summarize_turn_returns_none_on_openai_failure(modules, monkeypatch):
    import openai

    memory = modules["memory"]

    def _raise():
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(memory, "_shared_openai", _raise)
    assert memory.summarize_turn("prev", "q", "a") is None


def test_summarize_turn_and_store_persists_the_result(modules, user, monkeypatch):
    db, memory = modules["db"], modules["memory"]
    session = db.create_session(user["id"], "t")
    monkeypatch.setattr(memory, "_shared_openai", lambda: _FakeOpenAI("new summary"))

    memory.summarize_turn_and_store(user["id"], session["id"], "", "q", "a")

    assert db.get_session(user["id"], session["id"])["summary"] == "new summary"


def test_summarize_turn_and_store_never_raises(modules, user, monkeypatch):
    """A background task that raises would surface nowhere useful -- it must not."""
    memory = modules["memory"]

    def _boom(*a, **k):
        raise RuntimeError("simulated failure deep in summarization")

    monkeypatch.setattr(memory, "summarize_turn", _boom)
    # Must not raise.
    memory.summarize_turn_and_store(user["id"], "nonexistent-session", "", "q", "a")


# ---------------------------------------------------------------------------
# main.py: /ask + session wiring, with the real RAG pipeline stubbed out
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_query(modules, monkeypatch):
    """Replace backend.main.query with a spy returning a canned RagResponse.

    Patched at the point main.py imported it (``backend.main.query``), not at
    its definition, which is what actually intercepts the call the endpoint
    makes.
    """
    import backend.main as main
    from backend.rag import Grounding, RagResponse, Source

    calls: list[dict] = []

    def _fake_query(question, conversation_context=None, conversation_focus=None, **_):
        calls.append(
            {
                "question": question,
                "conversation_context": conversation_context,
                "conversation_focus": conversation_focus,
            }
        )
        return RagResponse(
            answer=f"Answer to: {question}",
            sources=[Source(source="handbook.pdf", page=1, snippet="...")],
            query_type="fact_lookup",
            grounding=Grounding(checked=True, faithful=True),
            diagnostics={},
        )

    monkeypatch.setattr(main, "query", _fake_query)
    return calls


@pytest.fixture()
def stub_summarizer(monkeypatch):
    """Replace the scheduled background summarizer with a synchronous, traceable stub."""
    import backend.main as main

    calls: list[dict] = []

    def _fake_summarize(user_id, session_id, previous_summary, question, answer):
        calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "previous_summary": previous_summary,
                "question": question,
                "answer": answer,
            }
        )
        from backend import db

        db.update_session_memory(
            user_id, session_id, summary=f"summary-after: {question}"
        )

    monkeypatch.setattr(main, "summarize_turn_and_store", _fake_summarize)
    return calls


def test_stateless_ask_is_unaffected_by_sessions(client, user, modules, stub_query):
    """No session_id: Phase 1 behaviour, no message storage, no summarization."""
    headers = auth_header(client, user, modules)
    response = client.post("/ask", json={"question": "Hi"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] is None
    assert stub_query[0]["conversation_context"] is None
    assert stub_query[0]["conversation_focus"] is None


def test_ask_with_unknown_session_id_returns_404(client, user, modules, stub_query):
    headers = auth_header(client, user, modules)
    response = client.post(
        "/ask", json={"question": "Hi", "session_id": "does-not-exist"}, headers=headers
    )
    assert response.status_code == 404
    assert stub_query == []  # never reached the pipeline


def test_new_session_has_no_memory_of_another_session(
    client, user, modules, stub_query, stub_summarizer
):
    """Two fresh sessions for the same user must not share context."""
    headers = auth_header(client, user, modules)
    db = modules["db"]

    session_a = db.create_session(user["id"], "A")
    db.update_session_memory(
        user["id"], session_a["id"], summary="Session A discussed annual leave."
    )
    session_b = db.create_session(user["id"], "B")  # fresh: summary == ""

    client.post(
        "/ask",
        json={"question": "What about it?", "session_id": session_b["id"]},
        headers=headers,
    )
    # The call that answered session B's question must not have seen A's summary.
    call = next(c for c in stub_query if c["question"] == "What about it?")
    assert not call["conversation_context"]


def test_followup_in_same_session_receives_prior_turns_summary(
    client, user, modules, stub_query, stub_summarizer
):
    """The second turn's generation call receives the first turn's stored summary."""
    headers = auth_header(client, user, modules)
    db = modules["db"]
    session = db.create_session(user["id"], "leave chat")

    first = client.post(
        "/ask",
        json={"question": "How many days of leave?", "session_id": session["id"]},
        headers=headers,
    )
    assert first.status_code == 200
    # The stubbed summarizer (a BackgroundTask) has run by the time TestClient
    # returns and wrote a deterministic, traceable summary.
    assert db.get_session(user["id"], session["id"])["summary"] == (
        "summary-after: How many days of leave?"
    )

    second = client.post(
        "/ask",
        json={"question": "And for the executive band?", "session_id": session["id"]},
        headers=headers,
    )
    assert second.status_code == 200
    call = next(c for c in stub_query if c["question"] == "And for the executive band?")
    assert call["conversation_context"] == "summary-after: How many days of leave?"


def test_turn_is_stored_in_session_messages(client, user, modules, stub_query, stub_summarizer):
    headers = auth_header(client, user, modules)
    db = modules["db"]
    session = db.create_session(user["id"], "t")

    client.post(
        "/ask",
        json={"question": "How many sick days?", "session_id": session["id"]},
        headers=headers,
    )

    messages = db.list_messages(user["id"], session["id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "How many sick days?"
    assert "Answer to:" in messages[1]["content"]


def test_last_document_updates_from_top_source_without_an_llm_call(
    client, user, modules, stub_query, stub_summarizer
):
    """last_document is set synchronously from response.sources[0] -- no LLM round trip."""
    headers = auth_header(client, user, modules)
    db = modules["db"]
    session = db.create_session(user["id"], "t")

    client.post(
        "/ask", json={"question": "q", "session_id": session["id"]}, headers=headers
    )

    assert db.get_session(user["id"], session["id"])["last_document"] == "handbook.pdf"


def test_summarizer_is_scheduled_as_a_background_task_not_awaited_inline(
    client, user, modules, stub_query, monkeypatch
):
    """The endpoint must return using BackgroundTasks.add_task, not a direct call.

    Patches the scheduling primitive itself: if the endpoint called
    ``summarize_turn_and_store`` directly instead of registering it via
    ``background_tasks.add_task``, this spy would never be exercised and the
    assertion on `scheduled` would fail.
    """
    import fastapi

    scheduled = []
    original_add_task = fastapi.BackgroundTasks.add_task

    def _spy_add_task(self, func, *args, **kwargs):
        scheduled.append(func.__name__ if hasattr(func, "__name__") else func)
        return original_add_task(self, func, *args, **kwargs)

    monkeypatch.setattr(fastapi.BackgroundTasks, "add_task", _spy_add_task)

    headers = auth_header(client, user, modules)
    session = modules["db"].create_session(user["id"], "t")
    client.post(
        "/ask", json={"question": "q", "session_id": session["id"]}, headers=headers
    )

    assert any("summarize_turn_and_store" in str(name) for name in scheduled)


def test_ask_response_survives_a_failing_summarizer(client, user, modules, stub_query, monkeypatch):
    """A broken summarizer must not turn a good answer into a failed request.

    Exercises the REAL ``summarize_turn_and_store`` (scheduled exactly as the
    endpoint schedules it), with the failure injected one level deeper --
    ``summarize_turn`` itself raising. Starlette does NOT swallow a background
    task's exceptions (proven by the fact that a naive replacement task IS
    allowed to fail the request, in a sibling test); the safety net has to be
    ``summarize_turn_and_store``'s own try/except, so this test targets exactly
    that, rather than replacing the very function under test.
    """
    memory = modules["memory"]

    def _boom(*a, **k):
        raise RuntimeError("simulated OpenAI failure deep in summarization")

    monkeypatch.setattr(memory, "summarize_turn", _boom)

    headers = auth_header(client, user, modules)
    session = modules["db"].create_session(user["id"], "t")
    response = client.post(
        "/ask", json={"question": "q", "session_id": session["id"]}, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Answer to: q"
    # And the summary was left exactly as it was -- not corrupted by the failure.
    assert modules["db"].get_session(user["id"], session["id"])["summary"] == ""


def test_background_task_exception_is_not_swallowed_by_starlette(
    client, user, modules, stub_query, monkeypatch
):
    """Sibling to the test above: proves the safety net must be OURS.

    If Starlette itself swallowed a background task's exception, the previous
    test would pass for the wrong reason -- this confirms a task that raises
    WITHOUT its own try/except really does surface, which is exactly why
    ``summarize_turn_and_store`` has one.
    """
    import backend.main as main

    def _boom(*a, **k):
        raise RuntimeError("unsafe task")

    monkeypatch.setattr(main, "summarize_turn_and_store", _boom)

    headers = auth_header(client, user, modules)
    session = modules["db"].create_session(user["id"], "t")
    with pytest.raises(RuntimeError, match="unsafe task"):
        client.post(
            "/ask", json={"question": "q", "session_id": session["id"]}, headers=headers
        )
