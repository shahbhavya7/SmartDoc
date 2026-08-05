"""SQLite schema and access layer for every relational entity in V2.

Chroma holds vectors and their metadata and nothing else. Users, documents,
chat sessions, and messages live here, because they need joins, uniqueness
constraints, and referential integrity that a vector store does not provide.

Design notes
------------
**Foreign keys are enforced, per connection.** SQLite compiles foreign-key
support in but leaves it OFF by default, so ``PRAGMA foreign_keys = ON`` is
issued on every connection -- declaring the constraint without the pragma looks
like integrity and provides none.

**Cascades are declared, not hand-rolled.** Deleting a user removes their
documents, sessions, and messages; deleting a session removes its messages. The
one thing SQLite cannot cascade into is Chroma, so document deletion is handled
explicitly in :func:`backend.documents.delete_document`.

**Every read takes a user_id.** There is deliberately no ``list_documents()``
without an owner argument -- the isolation gate is easiest to keep when the
unfiltered variant does not exist to be called by accident. The user_id passed
in always comes from a verified JWT.

**Ids are UUID4 strings, not autoincrement integers.** Sequential ids invite
enumeration from a client, and a string id lets the seeded dev user keep the
stable, readable id ``dev-user-0001``.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import backend.config as config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT,
    google_sub    TEXT UNIQUE,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    size_bytes      INTEGER,
    extraction_mode TEXT NOT NULL DEFAULT 'text',
    markdown_path   TEXT,
    UNIQUE (user_id, filename)
);

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title         TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    summary       TEXT NOT NULL DEFAULT '',
    last_document TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- V3.3 Layer C. One manifest row per document: the heading tree and the
-- aggregated topic/entity lists, both JSON because they are read whole. The
-- ITEMS below are a real table because enumeration routing queries them
-- structurally -- which is exactly the split the Chroma-stores-scalars rule forces.
CREATE TABLE IF NOT EXISTS document_manifests (
    document_id  TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL,
    source       TEXT NOT NULL,
    heading_tree TEXT NOT NULL DEFAULT '[]',
    topics       TEXT NOT NULL DEFAULT '[]',
    entities     TEXT NOT NULL DEFAULT '[]',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    table_count  INTEGER NOT NULL DEFAULT 0,
    item_count   INTEGER NOT NULL DEFAULT 0,
    built_at     TEXT NOT NULL
);

-- One row per enumerable thing the document exactly contains: a section, a table
-- row label, or a list item. This is what makes "there are 7" a fact rather than
-- a hope, and what an enumeration answer is checked against.
CREATE TABLE IF NOT EXISTS manifest_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('section', 'table_row', 'list_item')),
    group_label   TEXT NOT NULL DEFAULT '',
    -- Extra text a group can be NAMED by, beyond its label: a table's column
    -- headers. "list all record types" must find the group whose first column is
    -- "Record type" even though its section is called "Classification Tiers".
    group_context TEXT NOT NULL DEFAULT '',
    item          TEXT NOT NULL,
    item_norm     TEXT NOT NULL,
    heading_path  TEXT NOT NULL DEFAULT '',
    page          INTEGER,
    chunk_index   INTEGER,
    table_id      TEXT NOT NULL DEFAULT ''
);

-- Addendum 2. Every table cell, one row per cell (entity-attribute-value).
--
-- Long rather than wide because the tables being stored have no fixed shape: a
-- column-per-column schema would need DDL per document. EAV makes
-- "entity + column -> value" one indexed lookup and makes MAX/MIN/COUNT over a
-- column a plain aggregate, which is exactly the two access patterns this exists
-- for. ``value`` is TEXT even for numbers: it is the cell verbatim, and an answer
-- must be able to quote "78%" or "1,200" as the document wrote it. Numeric
-- comparison casts at query time instead.
--
-- source/page/table_title travel with every cell so a SQL-path answer can be
-- cited without a second lookup. An uncited exact value is a failure, not a
-- feature.
CREATE TABLE IF NOT EXISTS table_cells (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    table_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    table_title     TEXT NOT NULL DEFAULT '',
    page            INTEGER,
    row_index       INTEGER NOT NULL,
    row_entity      TEXT NOT NULL,
    row_entity_norm TEXT NOT NULL,
    column_name     TEXT NOT NULL,
    column_norm     TEXT NOT NULL,
    value           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user  ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_manifests_user ON document_manifests(user_id);
CREATE INDEX IF NOT EXISTS idx_manifest_items_user ON manifest_items(user_id, item_norm);
CREATE INDEX IF NOT EXISTS idx_manifest_items_doc ON manifest_items(document_id);
-- The lookup index. (user_id, row_entity_norm, column_norm) is the single-cell
-- path; the second index serves MAX/MIN/COUNT, which scan one column of one
-- table. user_id leads both: ownership is part of the key, not a filter applied
-- to rows the index already found.
CREATE INDEX IF NOT EXISTS idx_cells_lookup
    ON table_cells(user_id, row_entity_norm, column_norm);
CREATE INDEX IF NOT EXISTS idx_cells_column
    ON table_cells(user_id, column_norm, table_id);
CREATE INDEX IF NOT EXISTS idx_cells_doc ON table_cells(document_id);
"""

_init_lock = threading.Lock()
_initialised: set[str] = set()


class DBError(Exception):
    """Raised for constraint violations surfaced as application errors."""


class EmailAlreadyRegistered(DBError):
    """Signup attempted with an email that already has an account."""


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    return Path(config.SQLITE_PATH)


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> str:
    """ISO-8601 UTC timestamp. Stored as TEXT so it sorts lexicographically."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a connection with foreign keys on, committing on clean exit."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL keeps a long-running read (a chat listing) from blocking the
        # write that records the next message.
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the table's first release. ``CREATE TABLE IF NOT EXISTS``
# does not retrofit a table that already exists, so a database created under
# Phase 1 (before `summary`/`last_document` existed) needs an explicit ALTER.
_SESSION_MIGRATIONS = (
    ("summary", "TEXT NOT NULL DEFAULT ''"),
    ("last_document", "TEXT"),
)

# ``size_bytes`` is nullable rather than ``DEFAULT 0``: a row written before this
# column existed has an unknown size, and 0 would report it as empty. Callers
# resolve NULL from the stored file on disk instead (see backend/documents.py).
#
# V3.1 adds two more. ``extraction_mode`` defaults to 'text' rather than being
# nullable: every row that predates the column WAS ingested by the plain-text
# path, so 'text' is the true value, not a placeholder. ``markdown_path`` is
# nullable -- an absent markdown cache is genuinely unknown, not "".
_DOCUMENT_MIGRATIONS = (
    ("size_bytes", "INTEGER"),
    ("extraction_mode", "TEXT NOT NULL DEFAULT 'text'"),
    ("markdown_path", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, migrations in (
        ("sessions", _SESSION_MIGRATIONS),
        ("documents", _DOCUMENT_MIGRATIONS),
    ):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, ddl in migrations:
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db(force: bool = False) -> None:
    """Create the schema if absent. Cheap and idempotent; safe at import time."""
    path = str(_db_path())
    with _init_lock:
        if path in _initialised and not force:
            return
        with connect() as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)
        _initialised.add(path)


def reset_state_for_tests() -> None:
    """Forget which database files have been initialised (test support)."""
    with _init_lock:
        _initialised.clear()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def create_user(
    email: str,
    password_hash: str | None = None,
    google_sub: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Insert a user. Raises :class:`EmailAlreadyRegistered` on a duplicate.

    ``password_hash`` is None for a Google-only account and ``google_sub`` is
    None for a password account; a user may end up with both after linking.
    """
    record = {
        "id": user_id or new_id(),
        "email": email.strip().lower(),
        "password_hash": password_hash,
        "google_sub": google_sub,
        "created_at": utc_now(),
    }
    init_db()
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, google_sub, created_at)"
                " VALUES (:id, :email, :password_hash, :google_sub, :created_at)",
                record,
            )
    except sqlite3.IntegrityError as exc:
        raise EmailAlreadyRegistered(
            f"An account already exists for {email!r}."
        ) from exc
    return record


def get_user_by_email(email: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip().lower(),)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_google_sub(google_sub: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE google_sub = ?", (google_sub,)
        ).fetchone()
    return dict(row) if row else None


def link_google_sub(user_id: str, google_sub: str) -> None:
    """Attach a Google identity to an existing account (same verified email)."""
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE users SET google_sub = ? WHERE id = ?", (google_sub, user_id)
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# V3.3 Layer C -- the document manifest
# ---------------------------------------------------------------------------


def replace_manifest(
    user_id: str,
    document_id: str,
    source: str,
    heading_tree_json: str,
    topics_json: str,
    entities_json: str,
    items: list[dict],
    table_count: int = 0,
    chunk_count: int = 0,
) -> int:
    """Replace this document's manifest and items in one transaction.

    Replace, not upsert: a re-ingested document whose headings changed must not
    keep the old ones alongside the new. That is DECISIONS.md B1 one level up --
    surplus rows from a previous version surviving a shorter revision, and then
    being reported as present by the very component whose job is to say what
    exists.
    """
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM manifest_items WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM document_manifests WHERE document_id = ?", (document_id,))
        conn.execute(
            "INSERT INTO document_manifests (document_id, user_id, source,"
            " heading_tree, topics, entities, chunk_count, table_count, item_count,"
            " built_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_id, user_id, source, heading_tree_json, topics_json,
                entities_json, chunk_count, table_count, len(items), utc_now(),
            ),
        )
        conn.executemany(
            "INSERT INTO manifest_items (document_id, user_id, kind, group_label,"
            " group_context, item, item_norm, heading_path, page, chunk_index,"
            " table_id) VALUES (:document_id, :user_id, :kind, :group_label,"
            " :group_context, :item, :item_norm, :heading_path, :page,"
            " :chunk_index, :table_id)",
            [{**row, "document_id": document_id, "user_id": user_id} for row in items],
        )
    return len(items)


def get_manifest(user_id: str, document_id: str) -> dict | None:
    """One document's manifest row, or None. Scoped by owner."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_manifests WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def list_manifests(user_id: str) -> list[dict]:
    """Every manifest this user owns."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM document_manifests WHERE user_id = ? ORDER BY source",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def all_heading_paths(user_id: str) -> list[dict]:
    """Every distinct heading path this user's corpus contains.

    Used by the heading filter to decide, before any search runs, whether a
    question names a section that actually exists.
    """
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT heading_path, document_id FROM manifest_items"
            " WHERE user_id = ? AND heading_path <> ''",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def find_manifest_items(
    user_id: str,
    term: str,
    kinds: tuple[str, ...] = ("section", "table_row", "list_item"),
    limit: int = 300,
) -> list[dict]:
    """Manifest items whose item, group label, or group context contains ``term``.

    Ownership is in the WHERE clause, not a later filter -- the rule every other
    read in this system follows, and the reason an enumeration cannot accidentally
    enumerate another account's document.

    ``LIKE`` with an escaped pattern rather than FTS: these corpora are tens of
    documents, the index on ``(user_id, item_norm)`` covers the common case, and a
    full-text index would be one more thing to keep in step with re-ingestion.
    Word-boundary matching is applied by the caller, in one place.
    """
    if not term.strip():
        return []
    init_db()
    pattern = "%" + term.strip().lower().replace("%", r"\%").replace("_", r"\_") + "%"
    placeholders = ",".join("?" for _ in kinds)
    sql = (
        "SELECT * FROM manifest_items WHERE user_id = ?"
        f" AND kind IN ({placeholders})"
        r" AND (item_norm LIKE ? ESCAPE '\' OR lower(group_label) LIKE ? ESCAPE '\'"
        r" OR lower(group_context) LIKE ? ESCAPE '\')"
        " ORDER BY document_id, page, id LIMIT ?"
    )
    with connect() as conn:
        rows = conn.execute(
            sql, [user_id, *kinds, pattern, pattern, pattern, limit]
        ).fetchall()
    return [dict(r) for r in rows]


def manifest_group_items(
    user_id: str, document_id: str, group_label: str, kind: str
) -> list[dict]:
    """Every item in one group -- the authoritative "there are N of these" list."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM manifest_items WHERE user_id = ? AND document_id = ?"
            " AND group_label = ? AND kind = ? ORDER BY page, id",
            (user_id, document_id, group_label, kind),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Addendum 2 -- the relational table-cell store
# ---------------------------------------------------------------------------


def replace_table_cells(user_id: str, document_id: str, cells: list[dict]) -> int:
    """Replace this document's stored cells in one transaction.

    Replace, not upsert, for the reason ``replace_manifest`` gives one level up:
    a re-ingested spreadsheet whose row was deleted must not keep answering for
    that row. A stale exact value is worse than none -- it is stated
    authoritatively and is not in the document any more.
    """
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM table_cells WHERE document_id = ?", (document_id,))
        conn.executemany(
            "INSERT INTO table_cells (document_id, user_id, table_id, source,"
            " table_title, page, row_index, row_entity, row_entity_norm,"
            " column_name, column_norm, value) VALUES (:document_id, :user_id,"
            " :table_id, :source, :table_title, :page, :row_index, :row_entity,"
            " :row_entity_norm, :column_name, :column_norm, :value)",
            [{**row, "document_id": document_id, "user_id": user_id} for row in cells],
        )
    return len(cells)


def table_vocabulary(user_id: str) -> list[dict]:
    """Every distinct (table, column, entity) triple this user owns.

    Read once per user into the in-memory cache that Decision 1 fuzzy-matches
    against, so the fire/skip decision never costs a database round trip.
    """
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT table_id, source, table_title, column_name, column_norm,"
            " row_entity, row_entity_norm, MIN(page) AS page"
            " FROM table_cells WHERE user_id = ?"
            " GROUP BY table_id, column_norm, row_entity_norm"
            " ORDER BY table_id, row_index",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def lookup_cells(
    user_id: str, entity_norm: str, column_norm: str, limit: int = 10
) -> list[dict]:
    """Cells matching one resolved entity and one resolved column.

    Returns every match rather than the first: more than one IS the ambiguous
    case Decision 2 must see and discard, and a ``LIMIT 1`` here would hide it
    behind a confident-looking single answer.
    """
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM table_cells WHERE user_id = ? AND row_entity_norm = ?"
            " AND column_norm = ? ORDER BY table_id, row_index LIMIT ?",
            (user_id, entity_norm, column_norm, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def column_cells(
    user_id: str, column_norm: str, table_id: str | None = None, limit: int = 5000
) -> list[dict]:
    """Every cell in one column, for MAX/MIN/COUNT and threshold filters.

    Ranking happens in Python, not in SQL's ``ORDER BY CAST(value AS REAL)``:
    SQLite's cast silently yields 0.0 for "N/A" or "see note 3", which would make
    a non-numeric cell the confident winner of a MIN. The Python side parses and
    DISCARDS what it cannot read as a number, and reports how many it discarded.
    """
    init_db()
    sql = "SELECT * FROM table_cells WHERE user_id = ? AND column_norm = ?"
    params: list = [user_id, column_norm]
    if table_id:
        sql += " AND table_id = ?"
        params.append(table_id)
    sql += " ORDER BY table_id, row_index LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_table_cells(user_id: str) -> int:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM table_cells WHERE user_id = ?", (user_id,)
        ).fetchone()
    return int(row["n"]) if row else 0


def users_with_tables() -> list[str]:
    """Owners who have at least one stored cell, for warming the cache at boot."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT user_id FROM table_cells WHERE user_id <> ''"
        ).fetchall()
    return [r["user_id"] for r in rows]


def set_document_extraction(
    user_id: str, document_id: str, extraction_mode: str, markdown_path: str = ""
) -> None:
    """Record how a document was extracted (V3.1).

    Written after parsing rather than at row creation, because which path a PDF
    took is only known once conversion has been attempted -- and a scanned PDF
    that fell back must end up marked ``"text"``, not left at whatever the flag
    said it would be. Scoped by ``user_id`` so the update cannot touch another
    account's row even if a document id were guessed.
    """
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET extraction_mode = ?, markdown_path = ?"
            " WHERE id = ? AND user_id = ?",
            (extraction_mode or "text", markdown_path or None, document_id, user_id),
        )


def upsert_document(
    user_id: str,
    filename: str,
    document_id: str | None = None,
    size_bytes: int | None = None,
) -> dict:
    """Get-or-create this user's row for ``filename``, returning it.

    Re-uploading a filename must reuse the same ``document_id``: the id is
    stamped into every chunk's metadata, and minting a new one would orphan the
    previous version's vectors from the row that owns them.

    ``size_bytes`` is *updated* on an existing row rather than left alone, since
    a re-upload replaces the stored bytes -- keeping the first version's size
    would make the reported storage total drift from what is actually on disk.
    """
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        ).fetchone()
        if row:
            existing = dict(row)
            if size_bytes is not None and existing.get("size_bytes") != size_bytes:
                conn.execute(
                    "UPDATE documents SET size_bytes = ? WHERE id = ?",
                    (size_bytes, existing["id"]),
                )
                existing["size_bytes"] = size_bytes
            return existing
        record = {
            "id": document_id or new_id(),
            "user_id": user_id,
            "filename": filename,
            "created_at": utc_now(),
            "size_bytes": size_bytes,
        }
        conn.execute(
            "INSERT INTO documents (id, user_id, filename, created_at, size_bytes)"
            " VALUES (:id, :user_id, :filename, :created_at, :size_bytes)",
            record,
        )
    return record


def list_documents(user_id: str) -> list[dict]:
    """This user's documents, newest first. There is no unscoped variant."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC, filename",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(user_id: str, document_id: str) -> dict | None:
    """Fetch one document *owned by* ``user_id``.

    The owner is part of the lookup, not a check applied afterwards, so another
    user's id simply does not resolve -- the endpoint returns 404 and leaks
    nothing about whether that id exists.
    """
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def delete_document_row(user_id: str, document_id: str) -> bool:
    """Delete the owned row; False if it does not exist for this user."""
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id)
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Sessions and messages
# ---------------------------------------------------------------------------


def create_session(user_id: str, title: str = "") -> dict:
    init_db()
    record = {
        "id": new_id(),
        "user_id": user_id,
        "title": title,
        "created_at": utc_now(),
        "summary": "",
        "last_document": None,
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, created_at)"
            " VALUES (:id, :user_id, :title, :created_at)",
            record,
        )
    return record


def list_sessions(user_id: str, limit: int = 10) -> list[dict]:
    """This user's sessions, most recently ACTIVE first, for the chat sidebar.

    "Active" is the newest message time, not creation time -- a session started
    yesterday with a message five minutes ago belongs above one created an hour
    ago with no reply yet. Sessions with no messages fall back to their own
    ``created_at`` via ``COALESCE`` so a brand-new empty session still sorts
    sensibly (at the top, since it was just created).

    The join runs over ``idx_messages_session`` and groups by session id, which
    stays fast at this schema's scale without a dedicated "last activity" column
    to keep in sync on every message insert.
    """
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.*, COALESCE(MAX(m.created_at), s.created_at) AS last_activity "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "WHERE s.user_id = ? "
            "GROUP BY s.id "
            "ORDER BY last_activity DESC "
            "LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(user_id: str, session_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
    return dict(row) if row else None


def update_session_memory(
    user_id: str,
    session_id: str,
    summary: str | None = None,
    last_document: str | None = None,
) -> bool:
    """Update a session's running summary and/or last-discussed document.

    Ownership-checked like every other session write. ``summary`` replaces the
    stored value rather than appending -- the summariser is responsible for
    producing an already-condensed running summary, so storage does not also
    need append/truncate logic here.
    """
    if get_session(user_id, session_id) is None:
        return False
    sets, params = [], []
    if summary is not None:
        sets.append("summary = ?")
        params.append(summary)
    if last_document is not None:
        sets.append("last_document = ?")
        params.append(last_document)
    if not sets:
        return True
    params.extend([session_id, user_id])
    with connect() as conn:
        conn.execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id = ? AND user_id = ?",
            params,
        )
    return True


def delete_session(user_id: str, session_id: str) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        )
    return cur.rowcount > 0


def add_message(user_id: str, session_id: str, role: str, content: str) -> dict | None:
    """Append a message to a session ``user_id`` owns; None if they do not.

    Ownership is verified through the session rather than trusted from the
    caller, so a valid token cannot write into someone else's conversation.
    """
    if get_session(user_id, session_id) is None:
        return None
    record = {
        "id": new_id(),
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": utc_now(),
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at)"
            " VALUES (:id, :session_id, :role, :content, :created_at)",
            record,
        )
    return record


def list_messages(user_id: str, session_id: str) -> list[dict] | None:
    """Messages in a session ``user_id`` owns; None if they do not own it.

    The join on ``sessions.user_id`` is what enforces isolation: messages have
    no owner column of their own, so filtering on ``session_id`` alone would let
    a guessed id return another user's conversation.
    """
    init_db()
    with connect() as conn:
        owned = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
        if owned is None:
            return None
        rows = conn.execute(
            "SELECT m.* FROM messages m"
            " JOIN sessions s ON s.id = m.session_id"
            " WHERE m.session_id = ? AND s.user_id = ?"
            # Tie-broken by rowid (SQLite's implicit, monotonically increasing
            # insertion-order column), not m.id: two messages stored within the
            # same wall-clock SECOND -- e.g. a question and its answer, which
            # `created_at`'s second-resolution timestamp cannot tell apart --
            # would otherwise sort by UUID, which is effectively random and
            # silently reversed a real turn's order.
            " ORDER BY m.created_at, m.rowid",
            (session_id, user_id),
        ).fetchall()
    return [dict(r) for r in rows]
