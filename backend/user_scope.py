"""The single place that decides *whose* data a piece of work may touch.

Isolation is enforced at a choke point rather than at each call site. The
retrieval stack is final and spans six modules and ~4000 lines; threading a
``user_id`` parameter through every function in it would mean editing the very
code V2 is forbidden to modify, and -- worse -- one missed call site would be an
invisible cross-tenant leak. Instead the authenticated user is bound to a
:mod:`contextvars` context by the auth dependency, and every read of Chroma
funnels through :mod:`backend.vectorstore`, which asks this module for a filter.

Why a context variable and not a global: FastAPI runs sync endpoints in a
threadpool, and ``contextvars`` are copied into those workers, so concurrent
requests from different users cannot see each other's scope. A module-level
global would be shared across them.

The user_id placed here always originates from a verified JWT
(:mod:`backend.auth`). Nothing in this module accepts one from a client.

Id namespacing
--------------
Chunk ids were ``"<source>:<chunk_index>"``, which is unique per document but
NOT per user: two users uploading ``handbook.pdf`` would collide, and the second
upsert would silently overwrite the first user's chunks. Ids written under a
user scope are therefore prefixed ``"u<user_id>|"``, and the id-shaped metadata
fields (``id``, ``parent_id``, ``prev_id``, ``next_id``) are prefixed with it too
so neighbour and parent lookups keep resolving.

Chunks indexed before V2 carry unprefixed ids and unprefixed links. They are
internally consistent, so they keep working; the backfill only stamps
``user_id``/``document_id`` onto their metadata.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import backend.config as config

# None means "no scope bound" -- either the multi-user layer is off, or this is
# an internal maintenance task (ingest script, backfill) running outside a
# request. It never means "every user".
_current_user_id: ContextVar[str | None] = ContextVar(
    "smartdoc_current_user_id", default=None
)

ID_PREFIX_SEPARATOR = "|"


class ScopeError(Exception):
    """Raised when a scoped operation is attempted with no user bound."""


def set_current_user(user_id: str | None):
    """Bind ``user_id`` as the active scope. Returns the contextvars token."""
    return _current_user_id.set(user_id)


def reset_current_user(token) -> None:
    """Restore the scope that was active before :func:`set_current_user`."""
    _current_user_id.reset(token)


def current_user_id() -> str | None:
    """The user_id bound to this context, or None if unscoped."""
    if not config.MULTI_USER_ENABLED:
        return None
    return _current_user_id.get()


def require_user_id() -> str:
    """Like :func:`current_user_id` but refuses to proceed unscoped."""
    user_id = current_user_id()
    if not user_id:
        raise ScopeError(
            "No authenticated user is bound to this context. A scoped store "
            "operation was attempted outside a request; this is a bug, not a "
            "reason to fall back to unfiltered access."
        )
    return user_id


@contextmanager
def user_scope(user_id: str | None) -> Iterator[str | None]:
    """Run a block with ``user_id`` as the active scope."""
    token = set_current_user(user_id)
    try:
        yield user_id
    finally:
        reset_current_user(token)


# ---------------------------------------------------------------------------
# Chroma filters
# ---------------------------------------------------------------------------


def scoped_where(where: dict | None = None, user_id: str | None = None) -> dict | None:
    """AND ``where`` with a ``user_id`` equality filter.

    Chroma rejects a filter with two top-level keys, so the merge is expressed
    as ``$and`` rather than by mutating the caller's dict -- which also keeps the
    caller's filter untouched.
    """
    scope = user_id or current_user_id()
    if not scope:
        return where or None
    scope_clause = {"user_id": scope}
    if not where:
        return scope_clause
    if "$and" in where and len(where) == 1:
        return {"$and": [*where["$and"], scope_clause]}
    return {"$and": [where, scope_clause]}


def belongs_to_scope(metadata: dict | None) -> bool:
    """True if ``metadata`` is visible under the active scope.

    Used for post-filtering results from Chroma reads that cannot take a
    ``where`` clause (``get(ids=...)``), and as a defence-in-depth check.
    """
    scope = current_user_id()
    if not scope:
        return True
    return (metadata or {}).get("user_id") == scope


# ---------------------------------------------------------------------------
# Id namespacing
# ---------------------------------------------------------------------------


def id_prefix(user_id: str | None = None) -> str:
    """The namespace prefix for ids written under ``user_id`` (may be empty)."""
    scope = user_id or current_user_id()
    return f"u{scope}{ID_PREFIX_SEPARATOR}" if scope else ""


def scoped_id(raw_id: str, user_id: str | None = None) -> str:
    """Namespace ``raw_id``, leaving an already-prefixed id untouched."""
    prefix = id_prefix(user_id)
    if not prefix or not raw_id or raw_id.startswith(prefix):
        return raw_id
    return f"{prefix}{raw_id}"


# Metadata fields that hold references to other chunk ids and must be
# namespaced in step with the ids themselves.
LINK_FIELDS = ("id", "parent_id", "prev_id", "next_id")


def scope_metadata(meta: dict, document_id: str, user_id: str | None = None) -> dict:
    """Return ``meta`` stamped with ownership and with its id links namespaced.

    ``document_id`` is what makes the delete cascade possible: it is the join key
    back to the SQLite ``documents`` row, so deleting a document can find its
    vectors without relying on the filename.
    """
    scope = user_id or current_user_id()
    out = dict(meta)
    if document_id:
        out["document_id"] = document_id
    if not scope:
        return out
    out["user_id"] = scope
    for field in LINK_FIELDS:
        value = out.get(field)
        if value:
            out[field] = scoped_id(str(value), scope)
    return out
