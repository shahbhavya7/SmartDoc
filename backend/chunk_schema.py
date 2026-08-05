"""V3.3: the universal chunk metadata contract.

Every chunk written to Chroma carries **every** field below. Text chunks, table
parts, table summary chunks, and plain-text fallback chunks alike. Where a field
does not apply, an explicit default is written -- a field may be empty, it may
never be *absent*.

Why "may be empty but never absent" matters
-------------------------------------------
Absent keys are the reason metadata-driven code is normally written defensively:
every consumer ends up doing ``metadata.get("heading_path", "")`` and no consumer
can tell "this chunk has no heading" from "this chunk was written by an older path
that did not set the key". Filtering becomes unreliable, because a Chroma ``where``
clause on a key that only *some* chunks carry silently excludes the rest -- not as
an error, as a smaller result set. Guaranteeing presence makes a filter mean what
it says, and makes a missing key a bug that write-time validation catches instead
of a query that quietly under-returns.

Two consumers, deliberately separated
-------------------------------------
Rich metadata must not become a token tax. The fields split by who reads them:

* :data:`PROMPT_VISIBLE` -- a small set the *model* sees, so it can cite correctly
  and pick an output shape.
* :data:`FILTER_ONLY` -- read by *code*, for routing, filtering and expansion.
  These never enter the prompt. ``answers_questions`` and ``topics`` in particular
  are generated text (Layer B); keeping them out of the prompt is what makes "a
  wrong auto-tag can never corrupt an answer" structural rather than a promise.

Scalars only
------------
Chroma metadata values must be ``str``/``int``/``float``/``bool``. Lists are stored
as delimited strings; anything that needs to be *queried* structurally (the
manifest, per-document item lists) lives in SQLite instead.
"""

from __future__ import annotations

import logging

import backend.config as config

logger = logging.getLogger("smartdoc.chunk_schema")

# Delimiter for list-valued fields, used by Layer B and the table headers.
LIST_DELIMITER = ", "

CONTENT_TYPES = ("policy", "procedure", "table", "definition", "other")

# THE CONTRACT: field -> default written when the field does not apply.
#
# The default's TYPE is also the field's declared type, and is what validation
# coerces to -- an int field holding "" would break every ``where`` clause that
# compares it numerically.
CHUNK_SCHEMA: dict[str, object] = {
    # --- provenance (Layer A, structural, always exact) ---
    "source": "",
    "page": 0,
    "chunk_index": 0,
    "user_id": "",
    "heading_path": "",
    "section_title": "",
    # --- semantics (Layer B, LLM-derived at ingest, retrieval aid only) ---
    "content_type": "other",
    "topics": "",
    "entities": "",
    "answers_questions": "",
    # --- table structure (V3.2) ---
    "table_id": "",
    "table_part": 0,
    "table_total_parts": 0,
    "table_headers": "",
    # --- how the document was read (V3.1) ---
    "extraction_mode": "text",
}

REQUIRED_FIELDS: tuple[str, ...] = tuple(CHUNK_SCHEMA)

# Seen by the answer model. Deliberately four fields: enough to cite precisely and
# to choose a shape, small enough that the per-chunk header stays a line or two.
PROMPT_VISIBLE: tuple[str, ...] = ("source", "page", "heading_path", "content_type")

# Read by code only. Never rendered into a prompt.
FILTER_ONLY: tuple[str, ...] = tuple(f for f in REQUIRED_FIELDS if f not in PROMPT_VISIBLE)

# Fields the rest of the system also writes but which are not part of the
# contract -- they may be absent, and nothing filters on them.
OPTIONAL_FIELDS: tuple[str, ...] = (
    "doc_title", "page_end", "parent_id", "prev_id", "next_id", "has_table",
    "token_count", "content_hash", "document_id", "is_table_summary",
    "table_rows", "table_spans_pages", "table_fragments",
)


class SchemaError(ValueError):
    """Raised when a chunk violates the contract and strict mode is on."""


def _coerce(field: str, value) -> object:
    """Coerce ``value`` to the type declared by the field's default."""
    default = CHUNK_SCHEMA[field]
    if value is None:
        return default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        if isinstance(value, (list, tuple, set)):
            # Chroma cannot store a list. Joining rather than dropping keeps the
            # information; the delimiter is what every consumer splits on.
            return LIST_DELIMITER.join(str(v) for v in value if str(v).strip())
        return str(value)
    return value


def missing_fields(metadata: dict) -> list[str]:
    """Contract fields absent from ``metadata``. Empty values are fine; absent is not."""
    return [field for field in REQUIRED_FIELDS if field not in metadata]


def non_scalar_fields(metadata: dict) -> list[str]:
    """Keys whose value Chroma would reject."""
    return [
        key
        for key, value in metadata.items()
        if not isinstance(value, (str, int, float, bool))
    ]


def apply_defaults(metadata: dict | None = None, **overrides) -> dict:
    """Return ``metadata`` with every contract field present and correctly typed.

    Non-contract keys are passed through untouched -- ``parent_id``, ``page_end``
    and the rest are still needed, they simply are not guaranteed.
    """
    out = dict(metadata or {})
    out.update(overrides)
    for field, default in CHUNK_SCHEMA.items():
        out[field] = _coerce(field, out[field]) if field in out else default
    return out


def validate(metadata: dict, *, where: str = "chunk", strict: bool | None = None) -> list[str]:
    """Check one chunk against the contract. Returns the problems found.

    Called at WRITE time, which is the only place that can guarantee the invariant:
    a chunk is validated after ownership has been stamped and immediately before it
    reaches Chroma, so nothing can be indexed without the full schema.

    Logs by default and raises when strict. Logging is the production behaviour on
    purpose -- refusing to index a chunk because an optional label is missing would
    trade a complete index for a tidy one -- while tests and the verification script
    run strict, so a violation fails loudly there instead of scrolling past.
    """
    problems: list[str] = []
    absent = missing_fields(metadata)
    if absent:
        problems.append(f"missing fields: {', '.join(absent)}")
    bad = non_scalar_fields(metadata)
    if bad:
        problems.append(f"non-scalar values: {', '.join(bad)}")
    content_type = metadata.get("content_type")
    if content_type not in CONTENT_TYPES:
        problems.append(f"content_type not in {CONTENT_TYPES}: {content_type!r}")

    if problems:
        message = f"{where}: " + "; ".join(problems)
        if strict if strict is not None else config.CHUNK_SCHEMA_STRICT:
            raise SchemaError(message)
        logger.warning("Chunk schema violation -- %s", message)
    return problems


def prompt_metadata(metadata: dict) -> dict:
    """The subset the answer model is allowed to see."""
    return {field: metadata.get(field, CHUNK_SCHEMA[field]) for field in PROMPT_VISIBLE}


def split_list(value: str) -> list[str]:
    """Read a delimited scalar back as a list."""
    return [part.strip() for part in str(value or "").split(LIST_DELIMITER.strip()) if part.strip()]
