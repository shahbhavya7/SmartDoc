"""Phase 1 acceptance tests: auth, per-user isolation, and the delete cascade.

The isolation checks are the hard gate. They are written to fail loudly if the
scope is ever dropped from a read path, so they exercise the real store rather
than a mock: a temporary Chroma directory and a temporary SQLite file, with a
deterministic fake embedder standing in for OpenAI so the suite runs offline and
reproducibly. Everything else -- chunking, fusion, metadata, deletion -- is the
production code path.

The fake embedder is a hashed bag-of-words projection. It is not semantically
meaningful and is not meant to be: these tests assert *who* can see a chunk, not
*how well* it ranks. Retrieval quality is measured by the eval harness.

Run:
    python -m pytest tests/test_phase1_isolation.py -v
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re

import pytest

# Point every store at a temp location BEFORE backend.config is imported, since
# it reads the environment once at import time.
_TMP = None


@pytest.fixture(scope="session", autouse=True)
def _isolated_stores(tmp_path_factory):
    """Redirect SQLite and Chroma to a throwaway directory for the whole run."""
    global _TMP
    _TMP = tmp_path_factory.mktemp("smartdoc_phase1")
    os.environ["SQLITE_PATH"] = str(_TMP / "test.db")
    os.environ["CHROMA_DIR"] = str(_TMP / "chroma")
    os.environ["MULTI_USER_ENABLED"] = "true"
    os.environ["JWT_SECRET"] = "test-secret-not-the-placeholder"
    os.environ["OPENAI_API_KEY"] = "sk-test-not-used"
    os.environ["BCRYPT_ROUNDS"] = "4"  # keep signup/login fast in tests
    os.environ.setdefault("ENABLE_DOC_ROUTING", "false")

    import backend.config as config

    importlib.reload(config)
    yield


@pytest.fixture(scope="session")
def modules(_isolated_stores):
    """Import the backend after the environment is redirected."""
    import backend.config as config
    from backend import auth, db, documents, user_scope, vectorstore
    from backend import retrieval, routing

    db.reset_state_for_tests()
    db.init_db()
    return {
        "config": config,
        "auth": auth,
        "db": db,
        "documents": documents,
        "retrieval": retrieval,
        "routing": routing,
        "user_scope": user_scope,
        "vectorstore": vectorstore,
    }


# ---------------------------------------------------------------------------
# Deterministic offline embedder
# ---------------------------------------------------------------------------

EMBED_DIM = 64


def fake_embed(texts):
    """Hash each token into a fixed-width vector. Deterministic, offline."""
    vectors = []
    for text in texts:
        vector = [0.0] * EMBED_DIM
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            vector[digest[0] % EMBED_DIM] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


def make_chunks(source: str, texts: list[str]):
    """Build child Documents shaped exactly like the ingestion pipeline's."""
    from langchain.docstore.document import Document

    docs = []
    for index, text in enumerate(texts):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "doc_title": source.replace(".pdf", "").title(),
                    "section": f"Section {index}",
                    "page": index + 1,
                    "page_end": index + 1,
                    "chunk_index": index,
                    "parent_id": f"{source}#p{index}",
                    "prev_id": f"{source}:{index - 1}" if index else "",
                    "next_id": f"{source}:{index + 1}" if index < len(texts) - 1 else "",
                    "has_table": False,
                    "token_count": len(text.split()),
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Fixtures: two users, each with a document containing a unique secret token
# ---------------------------------------------------------------------------

ALICE_SECRET = "zephyrine"  # appears only in Alice's document
BOB_SECRET = "quangle"  # appears only in Bob's document


@pytest.fixture(scope="session")
def world(modules):
    """Alice and Bob, each owning one indexed document."""
    auth, db, vectorstore = modules["auth"], modules["db"], modules["vectorstore"]
    user_scope = modules["user_scope"]

    alice = db.create_user("alice@example.com", auth.hash_password("alice-password"))
    bob = db.create_user("bob@example.com", auth.hash_password("bob-password"))

    alice_doc = db.upsert_document(alice["id"], "alice_policy.pdf")
    bob_doc = db.upsert_document(bob["id"], "bob_policy.pdf")

    with user_scope.user_scope(alice["id"]):
        vectorstore.ingest_documents(
            make_chunks(
                "alice_policy.pdf",
                [
                    f"Alice annual leave entitlement is {ALICE_SECRET} days per year.",
                    f"The {ALICE_SECRET} allowance is reviewed each quarter by finance.",
                    # A third chunk WITHOUT the secret: BM25 assigns a term that
                    # appears in every document an IDF of zero, so a two-chunk
                    # corpus would score the secret at 0 and prove nothing.
                    "Travel bookings must be approved by a line manager in advance.",
                ],
            ),
            embed_fn=fake_embed,
            document_id=alice_doc["id"],
        )
    with user_scope.user_scope(bob["id"]):
        vectorstore.ingest_documents(
            make_chunks(
                "bob_policy.pdf",
                [f"Bob expense limit is {BOB_SECRET} pounds per trip."],
            ),
            embed_fn=fake_embed,
            document_id=bob_doc["id"],
        )

    return {
        "alice": alice,
        "bob": bob,
        "alice_doc": alice_doc,
        "bob_doc": bob_doc,
    }


@pytest.fixture(scope="session")
def client(modules, world):
    """A TestClient over the real app."""
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


def auth_header(client, email, password):
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


# ---------------------------------------------------------------------------
# 1. Auth: signup, login, and Google each issue a working JWT
# ---------------------------------------------------------------------------


def test_signup_issues_working_jwt(client):
    response = client.post(
        "/auth/signup", json={"email": "new@example.com", "password": "new-password-1"}
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "new@example.com"
    assert me.json()["auth_methods"] == ["password"]


def test_signup_rejects_duplicate_email(client):
    body = {"email": "dupe@example.com", "password": "another-password"}
    assert client.post("/auth/signup", json=body).status_code == 201
    second = client.post("/auth/signup", json=body)
    assert second.status_code == 400
    assert second.json()["error"]["type"] == "registration_failed"


def test_password_login_issues_working_jwt(client, world):
    headers = auth_header(client, "alice@example.com", "alice-password")
    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["id"] == world["alice"]["id"]


def test_login_failures_are_indistinguishable(client):
    unknown = client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "whatever-1"}
    )
    wrong = client.post(
        "/auth/login", json={"email": "alice@example.com", "password": "wrong-password"}
    )
    assert unknown.status_code == wrong.status_code == 401
    # Identical message: otherwise the endpoint enumerates registered addresses.
    assert unknown.json()["error"]["message"] == wrong.json()["error"]["message"]


def test_google_login_issues_working_jwt(client, modules):
    """The Google path mints a JWT from verified claims, like password login."""
    auth = modules["auth"]
    claims = {
        "sub": "google-oauth-sub-12345",
        "email": "google-user@example.com",
        "email_verified": True,
    }
    user = auth.upsert_google_user(claims)
    token = auth.create_access_token(user["id"], user["email"])

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "google-user@example.com"
    assert me.json()["auth_methods"] == ["google"]

    # Signing in again resolves the same account rather than creating another.
    assert auth.upsert_google_user(claims)["id"] == user["id"]


def test_google_will_not_hijack_a_password_account_on_unverified_email(client, modules):
    auth = modules["auth"]
    with pytest.raises(auth.AuthError):
        auth.upsert_google_user(
            {
                "sub": "attacker-sub-999",
                "email": "alice@example.com",
                "email_verified": False,
            }
        )


def test_protected_endpoints_reject_missing_and_forged_tokens(client, modules):
    import jwt

    assert client.get("/documents").status_code == 401
    assert client.post("/ask", json={"question": "hi"}).status_code == 401

    forged = jwt.encode(
        {"sub": "alice-id", "exp": 9999999999}, "wrong-signing-key", algorithm="HS256"
    )
    response = client.get("/documents", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthenticated"


# ---------------------------------------------------------------------------
# 2. THE ISOLATION GATE
# ---------------------------------------------------------------------------


def test_dense_retrieval_never_crosses_users(modules, world):
    """A query for Alice's exact secret, run as Bob, returns nothing of Alice's."""
    vectorstore, user_scope = modules["vectorstore"], modules["user_scope"]

    with user_scope.user_scope(world["bob"]["id"]):
        hits = vectorstore.query_collection(
            query_text=f"{ALICE_SECRET} annual leave entitlement",
            top_k=20,
            embed_fn=fake_embed,
        )
    sources = {h["metadata"]["source"] for h in hits}
    assert "alice_policy.pdf" not in sources
    assert all(h["metadata"]["user_id"] == world["bob"]["id"] for h in hits)
    assert not any(ALICE_SECRET in (h["document"] or "") for h in hits)


def test_keyword_search_never_crosses_users(modules, world):
    """BM25 is a materialised copy of the corpus, so it is checked separately."""
    retrieval, user_scope = modules["retrieval"], modules["user_scope"]

    with user_scope.user_scope(world["alice"]["id"]):
        alice_hits = retrieval.keyword_search(ALICE_SECRET, k=20)
    assert alice_hits, "Alice must be able to find her own rare token"

    with user_scope.user_scope(world["bob"]["id"]):
        bob_hits = retrieval.keyword_search(ALICE_SECRET, k=20)
    assert bob_hits == [], "Alice's rare token leaked into Bob's lexical index"


def test_corpus_listing_and_chunk_reads_are_scoped(modules, world):
    vectorstore, routing = modules["vectorstore"], modules["routing"]
    user_scope = modules["user_scope"]

    with user_scope.user_scope(world["bob"]["id"]):
        assert routing.corpus_documents() == ["bob_policy.pdf"]
        assert {c["metadata"]["source"] for c in vectorstore.all_chunks()} == {
            "bob_policy.pdf"
        }
        # Naming Alice's document explicitly does not conjure it.
        assert routing.document_chunks(["alice_policy.pdf"]) == []
        assert routing.document_outline("alice_policy.pdf") == []
        assert vectorstore.collection_stats()["count"] == 1


def test_chunk_reads_by_id_are_scoped(modules, world):
    """Guessing another user's chunk id returns nothing."""
    vectorstore, user_scope = modules["vectorstore"], modules["user_scope"]

    with user_scope.user_scope(world["alice"]["id"]):
        alice_ids = [c["id"] for c in vectorstore.all_chunks()]
    assert alice_ids

    with user_scope.user_scope(world["bob"]["id"]):
        assert vectorstore.get_chunks_by_ids(alice_ids) == []
        assert vectorstore.get_parents([f"u{world['alice']['id']}|alice_policy.pdf#p0"]) == {}


def test_same_filename_from_two_users_does_not_collide(modules, world):
    """Both users upload 'shared.pdf'; neither overwrites nor sees the other."""
    vectorstore, user_scope, db = (
        modules["vectorstore"],
        modules["user_scope"],
        modules["db"],
    )
    alice_doc = db.upsert_document(world["alice"]["id"], "shared.pdf")
    bob_doc = db.upsert_document(world["bob"]["id"], "shared.pdf")

    with user_scope.user_scope(world["alice"]["id"]):
        vectorstore.ingest_documents(
            make_chunks("shared.pdf", ["Alice version of the shared handbook."]),
            embed_fn=fake_embed,
            document_id=alice_doc["id"],
        )
    with user_scope.user_scope(world["bob"]["id"]):
        vectorstore.ingest_documents(
            make_chunks("shared.pdf", ["Bob version of the shared handbook."]),
            embed_fn=fake_embed,
            document_id=bob_doc["id"],
        )

    # Bob's write must not have replaced Alice's identically-named document.
    with user_scope.user_scope(world["alice"]["id"]):
        alice_shared = vectorstore.get_chunks_where({"source": "shared.pdf"})
    with user_scope.user_scope(world["bob"]["id"]):
        bob_shared = vectorstore.get_chunks_where({"source": "shared.pdf"})

    assert len(alice_shared) == 1 and "Alice version" in alice_shared[0]["document"]
    assert len(bob_shared) == 1 and "Bob version" in bob_shared[0]["document"]


def test_document_and_session_listings_are_scoped(client, world):
    bob = auth_header(client, "bob@example.com", "bob-password")

    listing = client.get("/documents", headers=bob).json()
    filenames = {d["filename"] for d in listing["documents"]}
    assert "alice_policy.pdf" not in filenames
    assert all(d["id"] != world["alice_doc"]["id"] for d in listing["documents"])

    # Alice's session is invisible to Bob, and its messages are unreadable.
    alice = auth_header(client, "alice@example.com", "alice-password")
    session = client.post("/sessions", json={"title": "Alice chat"}, headers=alice).json()
    client.post(
        f"/sessions/{session['id']}/messages",
        json={"role": "user", "content": f"What is the {ALICE_SECRET} allowance?"},
        headers=alice,
    )

    bob_sessions = client.get("/sessions", headers=bob).json()
    assert all(s["id"] != session["id"] for s in bob_sessions)

    assert client.get(f"/sessions/{session['id']}/messages", headers=bob).status_code == 404
    assert (
        client.post(
            f"/sessions/{session['id']}/messages",
            json={"role": "user", "content": "injected"},
            headers=bob,
        ).status_code
        == 404
    )
    # Alice still sees her own message -- the filter scopes, it does not blank.
    assert len(client.get(f"/sessions/{session['id']}/messages", headers=alice).json()) == 1


def test_no_endpoint_accepts_a_client_supplied_user_id(client, world):
    """Passing someone else's id as a body or query parameter changes nothing."""
    bob = auth_header(client, "bob@example.com", "bob-password")
    alice_id = world["alice"]["id"]

    listing = client.get(f"/documents?user_id={alice_id}", headers=bob).json()
    assert {d["filename"] for d in listing["documents"]} == {"bob_policy.pdf", "shared.pdf"}

    alice = auth_header(client, "alice@example.com", "alice-password")
    alice_session = client.post(
        "/sessions", json={"title": "Alice private"}, headers=alice
    ).json()

    # Asking for Alice's sessions while holding Bob's token yields Bob's.
    spoofed = client.get(f"/sessions?user_id={alice_id}", headers=bob).json()
    assert all(s["id"] != alice_session["id"] for s in spoofed)

    # A user_id in the *body* is ignored too: the session belongs to the token.
    created = client.post(
        "/sessions", json={"title": "t", "user_id": alice_id}, headers=bob
    ).json()
    assert all(s["id"] != created["id"] for s in client.get("/sessions", headers=alice).json())
    assert any(s["id"] == created["id"] for s in client.get("/sessions", headers=bob).json())


# ---------------------------------------------------------------------------
# 3. Delete cascade
# ---------------------------------------------------------------------------


def test_delete_removes_row_and_chunks_and_makes_doc_unretrievable(
    client, modules, world
):
    vectorstore, db, user_scope = (
        modules["vectorstore"],
        modules["db"],
        modules["user_scope"],
    )
    alice = auth_header(client, "alice@example.com", "alice-password")
    document_id = world["alice_doc"]["id"]

    with user_scope.user_scope(world["alice"]["id"]):
        before = vectorstore.collection_stats()["count"]
    assert before >= 3

    response = client.delete(f"/documents/{document_id}", headers=alice)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunks_deleted"] == 3
    assert body["row_deleted"] is True

    # (a) the SQLite row is gone
    assert db.get_document(world["alice"]["id"], document_id) is None
    assert all(
        d["id"] != document_id
        for d in client.get("/documents", headers=alice).json()["documents"]
    )

    # (b) the Chroma vector count dropped, and nothing of it is retrievable
    with user_scope.user_scope(world["alice"]["id"]):
        after = vectorstore.collection_stats()["count"]
        assert after == before - 3
        hits = vectorstore.query_collection(
            query_text=f"{ALICE_SECRET} annual leave", top_k=20, embed_fn=fake_embed
        )
        assert all(h["metadata"]["source"] != "alice_policy.pdf" for h in hits)
        assert vectorstore.get_chunks_where({"document_id": document_id}) == []
        # Parent records for the document went with it.
        assert vectorstore.get_parents(
            [f"u{world['alice']['id']}|alice_policy.pdf#p0"]
        ) == {}


def test_cannot_delete_another_users_document(client, modules, world):
    """Bob deleting Alice's id gets a 404 and destroys nothing."""
    vectorstore, user_scope = modules["vectorstore"], modules["user_scope"]
    bob = auth_header(client, "bob@example.com", "bob-password")

    with user_scope.user_scope(world["bob"]["id"]):
        bob_before = vectorstore.collection_stats()["count"]

    alice_docs = modules["db"].list_documents(world["alice"]["id"])
    assert alice_docs, "Alice should still own her shared.pdf"
    target = alice_docs[0]["id"]

    response = client.delete(f"/documents/{target}", headers=bob)
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"

    # Alice's document survived, and Bob's corpus is untouched.
    assert modules["db"].get_document(world["alice"]["id"], target) is not None
    with user_scope.user_scope(world["alice"]["id"]):
        assert vectorstore.get_chunks_where({"document_id": target})
    with user_scope.user_scope(world["bob"]["id"]):
        assert vectorstore.collection_stats()["count"] == bob_before
