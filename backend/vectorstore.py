"""Embedding + persistent vector storage.

Phase 2 scope only. This module takes Phase-1 ``Document`` chunks (from
``backend.ingestion``) and:

- embeds them with OpenAI ``text-embedding-3-small`` (``config.EMBED_MODEL``,
  the same model Phase 3 must use at query time so query and document
  vectors live in the same space), and
- writes them to a ChromaDB ``PersistentClient`` collection on disk at
  ``config.CHROMA_DIR``, storing ``{source, page, chunk_index}`` metadata
  alongside each vector.

It does not retrieve-for-answering, build prompts, or call a chat model --
those are later phases. It also does not fall back to a fake/local
embedding when ``OPENAI_API_KEY`` is missing: that would silently degrade
grounding quality, so the real path raises a clear error instead. Tests and
verification scripts may inject a different embedding function via the
``embed_fn`` parameter that every public function accepts.
"""

from __future__ import annotations

from typing import Callable, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from langchain.docstore.document import Document
from openai import OpenAI

from backend.config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL, OPENAI_API_KEY

EmbedFn = Callable[[list[str]], list[list[float]]]

# OpenAI's embeddings endpoint accepts a batch of inputs per request; batching
# keeps this well under request size / rate limits for typical corpora.
DEFAULT_EMBED_BATCH_SIZE = 100


class VectorStoreError(Exception):
    """Raised for configuration or embedding failures in this module."""


def _require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise VectorStoreError(
            "OPENAI_API_KEY is not set in .env. Set a real OpenAI API key "
            "before running ingestion or querying with the real embedding "
            "model -- this module does not silently fall back to a fake "
            "embedder."
        )


def openai_embed_fn(
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> EmbedFn:
    """Build an embedding function backed by the real OpenAI API.

    Args:
        batch_size: number of texts sent per embeddings API call.

    Returns:
        A function ``texts -> vectors`` using ``config.EMBED_MODEL``.

    Raises:
        VectorStoreError: if ``OPENAI_API_KEY`` is not configured. This is
            checked at call time (not just at import time) so the error
            surfaces exactly when an embedding is actually attempted.
    """

    def _embed(texts: list[str]) -> list[list[float]]:
        _require_api_key()
        client = OpenAI(api_key=OPENAI_API_KEY)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            # The API returns embeddings in the same order as the input list.
            vectors.extend(item.embedding for item in response.data)
        return vectors

    return _embed


def chunk_id(source: str, chunk_index: int) -> str:
    """Build a deterministic id for a chunk.

    Phase 1's ``chunk_index`` restarts at 0 for every document, so the id
    must combine ``source`` with ``chunk_index`` or ids collide across
    files. This id is what makes ingestion idempotent: re-embedding the
    same chunk of the same document always upserts the same row.
    """
    return f"{source}:{chunk_index}"


def get_client(persist_directory=CHROMA_DIR) -> chromadb.ClientAPI:
    """Return a ChromaDB ``PersistentClient`` writing to ``persist_directory``.

    This is the only client constructor used anywhere in the project --
    always disk-backed, never ``chromadb.Client()`` (in-memory) and never a
    plain Python list standing in for a vector index.
    """
    return chromadb.PersistentClient(path=str(persist_directory))


def get_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str = CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
) -> Collection:
    """Get-or-create the named collection on ``client`` (or a new one).

    Use this to load an already-ingested collection for querying without
    re-embedding anything -- no embedding function is required here because
    reads only need query-time embedding (see ``query_collection``), and
    writes go through ``upsert_documents``/``ingest_documents`` below.
    """
    client = client or get_client(persist_directory)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": EMBED_MODEL},
    )


def reset_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str = CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
) -> Collection:
    """Delete the named collection (if present) and recreate it empty.

    Used by the ``--reset`` clear-and-rebuild ingest path.
    """
    client = client or get_client(persist_directory)
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": EMBED_MODEL},
    )


def upsert_documents(
    documents: Sequence[Document],
    collection: Collection,
    embed_fn: EmbedFn | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> int:
    """Embed ``documents`` and upsert them into ``collection`` by deterministic id.

    Upserting by ``chunk_id(source, chunk_index)`` is what makes ingestion
    idempotent: re-running over the same corpus overwrites the same rows
    instead of duplicating them, and the embedding calls are batched
    (``embed_batch_size`` texts per API call) rather than one request per
    chunk.

    Args:
        documents: Phase-1 chunk ``Document`` objects, each with
            ``metadata`` containing ``source``, ``page``, ``chunk_index``.
        collection: a Chroma collection from ``get_collection``/``reset_collection``.
        embed_fn: embedding function to use; defaults to the real OpenAI
            path (``openai_embed_fn()``). Tests/verification scripts may
            inject a fake, deterministic embedder here instead.
        embed_batch_size: batch size forwarded to the default embed_fn.

    Returns:
        The number of documents upserted.
    """
    if not documents:
        return 0

    embed = embed_fn or openai_embed_fn(batch_size=embed_batch_size)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for doc in documents:
        source = doc.metadata["source"]
        chunk_index = doc.metadata["chunk_index"]
        page = doc.metadata["page"]
        ids.append(chunk_id(source, chunk_index))
        texts.append(doc.page_content)
        metadatas.append({"source": source, "page": page, "chunk_index": chunk_index})

    vectors = embed(texts)
    if len(vectors) != len(texts):
        raise VectorStoreError(
            f"Embedding function returned {len(vectors)} vectors for "
            f"{len(texts)} inputs -- these must match 1:1."
        )

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    return len(ids)


def ingest_documents(
    documents: Sequence[Document],
    collection_name: str = CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
    reset: bool = False,
    embed_fn: EmbedFn | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> int:
    """End-to-end: get/reset the collection, then upsert ``documents``.

    Args:
        documents: chunks to ingest.
        collection_name: Chroma collection name.
        persist_directory: on-disk directory for the PersistentClient.
        reset: if True, delete and recreate the collection first
            (clear-and-rebuild). If False (default), upsert by deterministic
            id so re-running does not duplicate rows.
        embed_fn: optional injected embedding function (see ``upsert_documents``).
        embed_batch_size: batch size forwarded to the default embed_fn.

    Returns:
        The number of documents upserted.
    """
    client = get_client(persist_directory)
    collection = (
        reset_collection(client, collection_name, persist_directory)
        if reset
        else get_collection(client, collection_name, persist_directory)
    )
    return upsert_documents(
        documents, collection, embed_fn=embed_fn, embed_batch_size=embed_batch_size
    )


def query_collection(
    query_text: str,
    collection: Collection | None = None,
    top_k: int | None = None,
    collection_name: str = CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
    embed_fn: EmbedFn | None = None,
) -> list[dict]:
    """Query an existing collection WITHOUT re-embedding the corpus.

    Only ``query_text`` is embedded here (one small call); the stored
    document vectors are read straight from disk. This is what a fresh
    process restart exercises: load the collection, embed the query, read
    back matches.

    Args:
        query_text: the natural-language query to embed and search with.
        collection: an already-loaded collection; if omitted, one is loaded
            via ``get_collection`` (which does not require re-embedding the
            corpus).
        top_k: number of results; defaults to ``config.TOP_K``.
        collection_name / persist_directory: used only if ``collection`` is
            not supplied.
        embed_fn: optional injected embedding function for the query; must
            match whatever embedded the stored documents, or distances are
            meaningless.

    Returns:
        A list of hits, each ``{"id", "document", "metadata", "distance"}``,
        ordered by increasing distance (most similar first).
    """
    from backend.config import TOP_K  # local import: keep config the source of truth

    collection = collection or get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    k = top_k or TOP_K
    embed = embed_fn or openai_embed_fn()
    query_vector = embed([query_text])[0]

    result = collection.query(query_embeddings=[query_vector], n_results=k)

    hits: list[dict] = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    for hit_id, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append({"id": hit_id, "document": doc, "metadata": meta, "distance": dist})
    return hits


def collection_stats(
    collection: Collection | None = None,
    collection_name: str = CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
) -> dict:
    """Return ``{"name", "count", "persist_directory"}`` for a collection."""
    collection = collection or get_collection(
        collection_name=collection_name, persist_directory=persist_directory
    )
    return {
        "name": collection.name,
        "count": collection.count(),
        "persist_directory": str(persist_directory),
    }
