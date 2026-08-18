"""Storage for ColPali page embeddings.

A SEPARATE SQLite database file (``colpali_store.db``), not a new table in
``smartdoc.db``. That is a stronger isolation guarantee than "additive table
in the same file": a bug in this module's schema or writes cannot corrupt or
lock the production database at all, and the whole experiment can be deleted
by removing one file.

Multi-vector embeddings (one variable-length array of patch vectors per
page) do not fit a single BLOB column cleanly, so each page's tensor is
serialized as raw float32 bytes plus its shape, and reconstructed with
``numpy.frombuffer(...).reshape(shape)`` on read. At experimental scale (a
handful of PDFs) everything is loaded into memory at query time -- no ANN
index, per the brief.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from colpali_experiment import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS colpali_page_embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    filename      TEXT NOT NULL,
    page_number   INTEGER NOT NULL,
    n_patches     INTEGER NOT NULL,
    embed_dim     INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    model_name    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (document_id, page_number, model_name)
);

CREATE INDEX IF NOT EXISTS idx_colpali_doc ON colpali_page_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_colpali_user ON colpali_page_embeddings(user_id);

-- Per-document ColPali readiness, written the instant /upload fans out
-- ingestion (status='pending') and updated by the background task when it
-- finishes (status='ready'/'failed'). Lives here, in this experiment's own
-- DB file, rather than as a column on the real `documents` table in
-- smartdoc.db -- adding a column there would violate the hard isolation
-- rule even though the column itself would be additive; a whole separate
-- table in a whole separate file is the stronger guarantee this experiment
-- has used everywhere else. document_id is the primary key: one status per
-- document, always the most recent ingestion attempt.
CREATE TABLE IF NOT EXISTS colpali_ingest_status (
    document_id  TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'failed')),
    error        TEXT,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_colpali_status_user ON colpali_ingest_status(user_id);
"""

# table_group_id: a page-level cluster id from the VISION-based page-to-page
# MaxSim continuity check (colpali_experiment/table_clustering.py). Deliberately
# its own column, added by migration rather than folded into the initial
# CREATE TABLE, following this project's own convention for post-release
# columns (see backend/db.py's _SESSION_MIGRATIONS/_DOCUMENT_MIGRATIONS). NULL
# means "not yet clustered" -- distinct from "" / -1, which would look like a
# cluster ID was computed and found nothing.
_EMBEDDING_MIGRATIONS = (
    ("table_group_id", "TEXT"),
)


def _db_path() -> Path:
    return config.COLPALI_DB_PATH


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(colpali_page_embeddings)")
    }
    for column, ddl in _EMBEDDING_MIGRATIONS:
        if column not in existing:
            conn.execute(f"ALTER TABLE colpali_page_embeddings ADD COLUMN {column} {ddl}")


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _serialize(embedding: np.ndarray) -> bytes:
    return np.ascontiguousarray(embedding, dtype=np.float32).tobytes()


def _deserialize(blob: bytes, n_patches: int, embed_dim: int) -> np.ndarray:
    # frombuffer's array is a read-only view over sqlite's returned bytes;
    # copied so downstream consumers (e.g. torch.from_numpy) get a writable,
    # independently-owned array rather than aliasing a buffer sqlite may reuse.
    return np.frombuffer(blob, dtype=np.float32).reshape(n_patches, embed_dim).copy()


def replace_page_embedding(
    document_id: str,
    user_id: str,
    filename: str,
    page_number: int,
    embedding: np.ndarray,
    model_name: str,
) -> None:
    """Insert or replace one page's embedding.

    Replace-not-append for the same reason every other store in this system
    replaces on re-ingestion (DECISIONS.md B1/S8): re-running the embedder on
    an unchanged document must not accumulate duplicate rows.
    """
    init_db()
    n_patches, embed_dim = embedding.shape
    with connect() as conn:
        conn.execute(
            "INSERT INTO colpali_page_embeddings (document_id, user_id, filename,"
            " page_number, n_patches, embed_dim, embedding_blob, model_name, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(document_id, page_number, model_name) DO UPDATE SET"
            " n_patches=excluded.n_patches, embed_dim=excluded.embed_dim,"
            " embedding_blob=excluded.embedding_blob, created_at=excluded.created_at",
            (
                document_id, user_id, filename, page_number,
                n_patches, embed_dim, _serialize(embedding), model_name, utc_now(),
            ),
        )


def delete_document_embeddings(document_id: str) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM colpali_page_embeddings WHERE document_id = ?", (document_id,)
        )
    return cur.rowcount


def get_document_embeddings(document_id: str) -> list[dict]:
    """Every stored page embedding for one document, in page order, as
    ``{"page_number", "embedding": np.ndarray, "n_patches", "embed_dim", ...}``.
    """
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM colpali_page_embeddings WHERE document_id = ?"
            " ORDER BY page_number",
            (document_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["embedding"] = _deserialize(d["embedding_blob"], d["n_patches"], d["embed_dim"])
        del d["embedding_blob"]
        result.append(d)
    return result


def get_user_embeddings(user_id: str) -> list[dict]:
    """Every stored page embedding this user owns, across all documents."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM colpali_page_embeddings WHERE user_id = ?"
            " ORDER BY document_id, page_number",
            (user_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["embedding"] = _deserialize(d["embedding_blob"], d["n_patches"], d["embed_dim"])
        del d["embedding_blob"]
        result.append(d)
    return result


def set_table_group(document_id: str, page_number: int, table_group_id: str | None) -> None:
    """Record which visual table-continuity cluster a page belongs to.

    Written by ``table_clustering.py`` only, never by the embedder -- keeping
    the writer of the cluster assignment separate from the writer of the
    embedding is what lets clustering be re-run (e.g. after changing the
    threshold) without re-embedding anything.
    """
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE colpali_page_embeddings SET table_group_id = ?"
            " WHERE document_id = ? AND page_number = ?",
            (table_group_id, document_id, page_number),
        )


def stats() -> dict:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pages, COUNT(DISTINCT document_id) AS documents"
            " FROM colpali_page_embeddings"
        ).fetchone()
    return {"pages": row["pages"], "documents": row["documents"]}


def set_ingest_status(
    document_id: str,
    user_id: str,
    filename: str,
    status: str,
    error: str | None = None,
) -> None:
    """Record this document's ColPali readiness -- 'pending'/'ready'/'failed'.

    One row per document (``document_id`` is the primary key), so re-ingesting
    (e.g. a re-upload) simply overwrites the previous attempt's outcome rather
    than accumulating history a status check would have to sort through.
    """
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO colpali_ingest_status"
            " (document_id, user_id, filename, status, error, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(document_id) DO UPDATE SET"
            " user_id=excluded.user_id, filename=excluded.filename,"
            " status=excluded.status, error=excluded.error,"
            " updated_at=excluded.updated_at",
            (document_id, user_id, filename, status, error, utc_now()),
        )


def get_ingest_status(document_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM colpali_ingest_status WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def get_ingest_statuses_for_user(user_id: str) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM colpali_ingest_status WHERE user_id = ?", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]
