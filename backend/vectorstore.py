"""Embedding, persistent vector storage, and the parent-chunk store.

Child chunks are embedded with ``config.EMBED_MODEL`` and written to a ChromaDB
``PersistentClient`` collection on disk. Parent chunks are not embedded (nothing
searches them directly) and are kept in a JSON sidecar next to the Chroma store,
keyed by ``parent_id``.

Correctness properties this module is responsible for
----------------------------------------------------
**Replacing a document actually replaces it.** The original implementation
upserted by ``source:chunk_index`` only. Re-ingesting an *edited* document that
produced fewer chunks than before left the surplus chunks of the old version in
the index permanently -- so a shortened or superseded policy stayed retrievable
and citable as if current. Proven with a controlled test: a 5-chunk document
re-ingested as 2 chunks left count at 5, with 3 stale chunks still live.
``ingest_documents`` now deletes every existing chunk for a source before
writing the new ones.

**The query model always matches the index model.** The collection records the
embedding model that built it, and ``assert_embedding_model`` refuses to query a
collection built by a different one. Without this, switching to another model of
the same dimensionality (e.g. ada-002, also 1536) returns confident nonsense
rather than an error.

**The distance metric is explicit.** Chroma defaults to ``l2``; the collection is
created with ``config.CHROMA_SPACE`` (cosine) so distances live on a stated 0-2
scale rather than an inherited 0-4 one.

**Unchanged documents are skipped.** Each chunk carries the document's
``content_hash``; ``needs_reingest`` compares it against what is indexed so
re-running ingestion over an unchanged corpus costs no embedding calls.

**No read escapes the active user's scope (V2).** This module is the only door
between the pipeline and Chroma, so it is where per-user isolation is applied:
every read merges ``{"user_id": <scope>}`` into its filter, and every write
stamps ``user_id``/``document_id`` onto the metadata and namespaces the chunk id.
The scope is set from a verified JWT by :mod:`backend.user_scope`; callers pass
no user_id and cannot opt out of the filter. With no scope bound (the ingest
scripts, or ``MULTI_USER_ENABLED=false``) behaviour is exactly as it was in V1.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Iterable, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from langchain.docstore.document import Document
from openai import OpenAI

import backend.config as config
from backend.chunk_schema import apply_defaults, validate
from backend.user_scope import (
    belongs_to_scope,
    current_user_id,
    scope_metadata,
    scoped_id,
    scoped_where,
)

EmbedFn = Callable[[list[str]], list[list[float]]]

DEFAULT_EMBED_BATCH_SIZE = 100

# Chroma rejects None and non-scalar metadata values.
_SCALARS = (str, int, float, bool)

_client_lock = threading.Lock()
_clients: dict[str, chromadb.ClientAPI] = {}
_openai_client: OpenAI | None = None


class VectorStoreError(Exception):
    """Raised for configuration, consistency, or embedding failures."""


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _require_api_key() -> None:
    if not config.OPENAI_API_KEY:
        raise VectorStoreError(
            "OPENAI_API_KEY is not set in .env. Set a real OpenAI API key before "
            "running ingestion or querying -- this module does not silently fall "
            "back to a fake embedder."
        )


def _shared_openai() -> OpenAI:
    """Return a process-wide OpenAI client.

    Constructing a client per call opens a fresh connection pool each time;
    reusing one keeps TLS handshakes out of the per-query latency budget.

    A request timeout and retry budget are set explicitly. The SDK's default is
    to wait indefinitely, so one hung connection blocks its caller forever with
    no error -- observed as an evaluation run that sat silent for 28 minutes, and
    the same hang would tie up a FastAPI worker in production.
    """
    global _openai_client
    if _openai_client is None:
        _require_api_key()
        _openai_client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
            max_retries=config.REQUEST_MAX_RETRIES,
        )
    return _openai_client


def openai_embed_fn(batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> EmbedFn:
    """Build an embedding function backed by the real OpenAI API."""

    def _embed(texts: list[str]) -> list[list[float]]:
        _require_api_key()
        client = _shared_openai()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
            # The API returns embeddings in input order.
            vectors.extend(item.embedding for item in response.data)
        return vectors

    return _embed


# ---------------------------------------------------------------------------
# Ids and client
# ---------------------------------------------------------------------------


def chunk_id(source: str, chunk_index: int) -> str:
    """Build a deterministic child-chunk id.

    ``chunk_index`` restarts at 0 for each document, so the id must combine it
    with ``source`` or ids collide across files.
    """
    return f"{source}:{chunk_index}"


def get_client(persist_directory=None) -> chromadb.ClientAPI:
    """Return a cached ``PersistentClient`` for ``persist_directory``.

    Always disk-backed -- never ``chromadb.Client()`` (in-memory). Clients are
    cached per directory because constructing one re-opens sqlite and reloads
    HNSW segments, which is wasted work on every request.
    """
    path = str(persist_directory or config.CHROMA_DIR)
    with _client_lock:
        if path not in _clients:
            # Telemetry disabled explicitly rather than via environment
            # variable: the installed posthog is incompatible with chromadb's
            # telemetry call, and the env var alone is not honoured by this
            # version, so every operation would print a capture() warning.
            _clients[path] = chromadb.PersistentClient(
                path=path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
        return _clients[path]


def _collection_metadata() -> dict:
    """Metadata stamped on collection creation."""
    return {
        "embedding_model": config.EMBED_MODEL,
        "hnsw:space": config.CHROMA_SPACE,
    }


def get_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str | None = None,
    persist_directory=None,
) -> Collection:
    """Get-or-create the named collection.

    No embedding function is attached: writes embed explicitly via
    ``upsert_documents`` and reads embed only the query, so loading a collection
    never risks re-embedding the corpus.
    """
    client = client or get_client(persist_directory)
    return client.get_or_create_collection(
        name=collection_name or config.CHROMA_COLLECTION,
        metadata=_collection_metadata(),
    )


def reset_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str | None = None,
    persist_directory=None,
) -> Collection:
    """Delete the named collection (if present) and recreate it empty."""
    client = client or get_client(persist_directory)
    name = collection_name or config.CHROMA_COLLECTION
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    _parent_path(persist_directory).unlink(missing_ok=True)
    return client.get_or_create_collection(name=name, metadata=_collection_metadata())


def assert_embedding_model(collection: Collection) -> None:
    """Fail loudly if the collection was built by a different embed model.

    Two 1536-dimensional models produce vectors that are geometrically
    compatible but semantically unrelated, so a mismatch cannot be detected from
    the data -- only from this recorded stamp.
    """
    recorded = (collection.metadata or {}).get("embedding_model")
    if recorded and recorded != config.EMBED_MODEL:
        raise VectorStoreError(
            f"Collection '{collection.name}' was built with embedding model "
            f"'{recorded}' but EMBED_MODEL is now '{config.EMBED_MODEL}'. Query "
            "and document vectors would be incomparable. Re-ingest with --reset, "
            "or restore the original EMBED_MODEL."
        )


# ---------------------------------------------------------------------------
# Parent store
# ---------------------------------------------------------------------------


def _parent_path(persist_directory=None) -> Path:
    """Path of the JSON sidecar holding parent chunks."""
    return Path(persist_directory or config.CHROMA_DIR) / "parents.json"


def _load_parents(persist_directory=None) -> dict[str, dict]:
    path = _parent_path(persist_directory)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt sidecar must not break retrieval; parent expansion degrades
        # to child-only context, which is still correct.
        return {}


def _write_parents(parents: dict[str, dict], persist_directory=None) -> None:
    path = _parent_path(persist_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parents), encoding="utf-8")


def save_parents(
    documents: Sequence[Document], persist_directory=None, document_id: str = ""
) -> int:
    """Store parent chunks, replacing this user's existing parents of the same sources.

    The purge is scoped by owner as well as by source: two users may both have a
    ``handbook.pdf``, and an unscoped purge would delete the other user's parents
    while re-indexing your own document.
    """
    scope = current_user_id()
    store = _load_parents(persist_directory)
    sources = {d.metadata["source"] for d in documents}
    store = {
        pid: rec
        for pid, rec in store.items()
        if not (rec.get("source") in sources and rec.get("user_id") == scope)
    }
    for doc in documents:
        record = {
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "doc_title": doc.metadata.get("doc_title", ""),
            "section": doc.metadata.get("section", ""),
            "page": doc.metadata.get("page"),
            "page_end": doc.metadata.get("page_end"),
            # V3.1. "" on the plain-text path, so a flag-OFF parent record is
            # unchanged in value.
            "heading_path": doc.metadata.get("heading_path", ""),
        }
        if scope:
            record["user_id"] = scope
        if document_id:
            record["document_id"] = document_id
        store[scoped_id(doc.metadata["id"])] = record
    _write_parents(store, persist_directory)
    return len(documents)


def get_parents(parent_ids: Iterable[str], persist_directory=None) -> dict[str, dict]:
    """Fetch parent records by id, restricted to the active user's parents.

    Parent ids are already namespaced per user, so a cross-user id would not
    match; the ownership check is a second, explicit barrier rather than a
    reliance on ids being unguessable.
    """
    store = _load_parents(persist_directory)
    scope = current_user_id()
    out: dict[str, dict] = {}
    for pid in parent_ids:
        record = store.get(pid)
        if record is None:
            continue
        if scope and record.get("user_id") != scope:
            continue
        out[pid] = record
    return out


def delete_parents_for_document(document_id: str, persist_directory=None) -> int:
    """Drop every parent record belonging to ``document_id`` (scoped to owner)."""
    scope = current_user_id()
    store = _load_parents(persist_directory)
    keep = {
        pid: rec
        for pid, rec in store.items()
        if not (
            rec.get("document_id") == document_id
            and (not scope or rec.get("user_id") == scope)
        )
    }
    removed = len(store) - len(keep)
    if removed:
        _write_parents(keep, persist_directory)
    return removed


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _clean_metadata(meta: dict) -> dict:
    """Drop None values and stringify anything Chroma cannot store."""
    out: dict = {}
    for key, value in meta.items():
        if value is None:
            continue
        out[key] = value if isinstance(value, _SCALARS) else str(value)
    return out


def indexed_hashes(
    collection: Collection | None = None, persist_directory=None
) -> dict[str, str]:
    """Return ``{source: content_hash}`` for the active user's indexed chunks."""
    collection = collection or get_collection(persist_directory=persist_directory)
    got = collection.get(include=["metadatas"], where=scoped_where())
    hashes: dict[str, str] = {}
    for meta in got.get("metadatas") or []:
        source = meta.get("source")
        digest = meta.get("content_hash")
        if source and digest:
            hashes[source] = digest
    return hashes


def needs_reingest(
    source: str, content_hash: str, collection: Collection | None = None
) -> bool:
    """Return True if ``source`` is absent or its indexed content differs."""
    return indexed_hashes(collection).get(source) != content_hash


def delete_source(
    source: str, collection: Collection | None = None, persist_directory=None
) -> int:
    """Delete the active user's indexed chunks for ``source``.

    This is what makes replacement correct rather than merely idempotent:
    without it, an edited document that yields fewer chunks leaves its surplus
    old chunks live in the index forever. Under a user scope the deletion is
    filtered by owner, so replacing your ``handbook.pdf`` cannot delete mine.
    """
    collection = collection or get_collection(persist_directory=persist_directory)
    ids = collection.get(where=scoped_where({"source": source})).get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def delete_document_chunks(
    document_id: str, collection: Collection | None = None, persist_directory=None
) -> int:
    """Delete every chunk of ``document_id``, and its parents. Returns the count.

    Matched on the ``document_id`` metadata stamped at ingest rather than on the
    filename, so a document deleted here cannot be confused with a same-named
    document belonging to anyone else.
    """
    collection = collection or get_collection(persist_directory=persist_directory)
    ids = collection.get(where=scoped_where({"document_id": document_id})).get("ids") or []
    if ids:
        collection.delete(ids=ids)
    delete_parents_for_document(document_id, persist_directory)
    return len(ids)


def get_chunks_by_ids(
    ids: Sequence[str], collection: Collection | None = None, persist_directory=None
) -> list[dict]:
    """Fetch chunks by id, dropping any the active user does not own.

    Chroma's ``get(ids=...)`` takes no ``where`` clause, so ownership is applied
    to the results. Callers that follow ``prev_id``/``next_id``/``parent_id``
    links go through here rather than touching the collection directly.
    """
    if not ids:
        return []
    collection = collection or get_collection(persist_directory=persist_directory)
    got = collection.get(ids=list(ids), include=["documents", "metadatas"])
    return [
        {"id": cid, "document": doc, "metadata": meta}
        for cid, doc, meta in zip(
            got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
        )
        if belongs_to_scope(meta)
    ]


def get_chunks_where(
    where: dict | None = None,
    collection: Collection | None = None,
    persist_directory=None,
    include: list[str] | None = None,
) -> list[dict]:
    """Fetch chunks matching ``where``, ANDed with the active user's scope."""
    collection = collection or get_collection(persist_directory=persist_directory)
    got = collection.get(
        where=scoped_where(where), include=include or ["documents", "metadatas"]
    )
    documents = got.get("documents") or [None] * len(got.get("ids") or [])
    return [
        {"id": cid, "document": doc, "metadata": meta}
        for cid, doc, meta in zip(got.get("ids") or [], documents, got.get("metadatas") or [])
    ]


def upsert_documents(
    documents: Sequence[Document],
    collection: Collection,
    embed_fn: EmbedFn | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    document_id: str = "",
) -> int:
    """Embed ``documents`` and upsert them by deterministic, user-namespaced id.

    Under a user scope, ids are prefixed with the owner and the id-shaped
    metadata links are prefixed to match. Without the prefix, two users
    uploading the same filename would produce identical ids and the second
    upsert would overwrite the first user's chunks.
    """
    if not documents:
        return 0

    embed = embed_fn or openai_embed_fn(batch_size=embed_batch_size)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for doc in documents:
        raw_id = chunk_id(doc.metadata["source"], doc.metadata["chunk_index"])
        ids.append(scoped_id(raw_id))
        texts.append(doc.page_content)
        # V3.3: the schema contract is enforced HERE -- after ownership has been
        # stamped and immediately before the chunk reaches Chroma. This is the only
        # point that can guarantee the invariant, because `user_id` does not exist
        # until scope_metadata has run. apply_defaults fills anything absent so a
        # violation cannot produce an unindexable chunk; validate() reports it, and
        # raises when CHUNK_SCHEMA_STRICT is on (tests and the verifier).
        stamped = apply_defaults(
            _clean_metadata(scope_metadata(doc.metadata, document_id=document_id))
        )
        validate(stamped, where=f"{stamped.get('source')}#{stamped.get('chunk_index')}")
        metadatas.append(stamped)

    vectors = embed(texts)
    if len(vectors) != len(texts):
        raise VectorStoreError(
            f"Embedding function returned {len(vectors)} vectors for {len(texts)} "
            "inputs -- these must match 1:1."
        )

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    return len(ids)


def ingest_documents(
    documents: Sequence[Document],
    parents: Sequence[Document] = (),
    collection_name: str | None = None,
    persist_directory=None,
    reset: bool = False,
    embed_fn: EmbedFn | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    replace_sources: bool = True,
    document_id: str = "",
) -> int:
    """Index child chunks (and store their parents), replacing prior versions.

    Args:
        replace_sources: delete all existing chunks of each incoming source
            before writing. Leave True unless deliberately adding to a source
            incrementally -- turning it off reintroduces the stale-chunk bug.
        document_id: the SQLite ``documents.id`` these chunks belong to, stamped
            into every chunk's metadata so deletion can find them by id.
    """
    client = get_client(persist_directory)
    collection = (
        reset_collection(client, collection_name, persist_directory)
        if reset
        else get_collection(client, collection_name, persist_directory)
    )

    if replace_sources and not reset:
        for source in {d.metadata["source"] for d in documents}:
            delete_source(source, collection)

    indexed = upsert_documents(
        documents,
        collection,
        embed_fn=embed_fn,
        embed_batch_size=embed_batch_size,
        document_id=document_id,
    )
    if parents:
        save_parents(parents, persist_directory, document_id=document_id)
    return indexed


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_collection(
    query_text: str | None = None,
    collection: Collection | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
    persist_directory=None,
    embed_fn: EmbedFn | None = None,
    query_vector: list[float] | None = None,
    where: dict | None = None,
) -> list[dict]:
    """Query an existing collection without re-embedding the corpus.

    Args:
        query_vector: a pre-computed query embedding, so callers can reuse one
            embedding across several searches.
        where: optional Chroma metadata filter (used by document routing). It is
            ANDed with the active user's scope, which the caller cannot remove.

    Returns:
        Hits as ``{"id", "document", "metadata", "distance"}``, nearest first.
    """
    collection = collection or get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    assert_embedding_model(collection)

    if query_vector is None:
        if not query_text:
            raise VectorStoreError("query_collection needs query_text or query_vector.")
        embed = embed_fn or openai_embed_fn()
        query_vector = embed([query_text])[0]

    k = top_k or config.TOP_K
    # Asking for more results than the collection holds raises in some Chroma
    # versions; clamp so a small corpus behaves like a large one.
    k = max(1, min(k, collection.count() or 1))

    effective_where = scoped_where(where)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        **({"where": effective_where} if effective_where else {}),
    )

    hits: list[dict] = []
    for hit_id, doc, meta, dist in zip(
        result.get("ids", [[]])[0],
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0],
    ):
        hits.append({"id": hit_id, "document": doc, "metadata": meta, "distance": dist})
    return hits


def all_chunks(
    collection: Collection | None = None, persist_directory=None
) -> list[dict]:
    """Return the active user's indexed chunks as ``{"id", "document", "metadata"}``.

    Used to build the in-memory keyword index. Scoping here is what keeps BM25
    single-tenant: the lexical index is built from this list, so an unfiltered
    read would let a rare identifier in someone else's document surface as a
    keyword hit even though dense search never saw it.
    """
    collection = collection or get_collection(persist_directory=persist_directory)
    got = collection.get(include=["documents", "metadatas"], where=scoped_where())
    return [
        {"id": cid, "document": doc, "metadata": meta}
        for cid, doc, meta in zip(
            got.get("ids") or [],
            got.get("documents") or [],
            got.get("metadatas") or [],
        )
    ]


def collection_stats(
    collection: Collection | None = None,
    collection_name: str | None = None,
    persist_directory=None,
) -> dict:
    """Return name, count, directory, embedding model, and document count.

    Under a user scope the counts describe *that user's* corpus, not the whole
    collection: a global chunk total would tell a signed-in user how much other
    people have uploaded.
    """
    collection = collection or get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    scope = scoped_where()
    metadatas = (
        collection.get(include=["metadatas"], where=scope).get("metadatas") or []
    )
    sources = {meta.get("source") for meta in metadatas}
    count = len(metadatas) if scope else collection.count()
    return {
        "name": collection.name,
        "count": count,
        "documents": len([s for s in sources if s]),
        "embedding_model": (collection.metadata or {}).get(
            "embedding_model", config.EMBED_MODEL
        ),
        "distance_space": (collection.metadata or {}).get(
            "hnsw:space", config.CHROMA_SPACE
        ),
        "persist_directory": str(persist_directory or config.CHROMA_DIR),
    }
