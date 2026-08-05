"""V3.3 Layer B: semantic chunk labels, extracted once at ingest.

What this buys
--------------
Dense retrieval matches a question against a chunk's *wording*. A chunk that
answers "how do I claim travel costs?" without using the words "claim" or
"travel" is hard to find, and no amount of ``k`` fixes vocabulary mismatch. So
each chunk is asked, once, what it is about: what kind of content it is, its
topics, the entities it names, and the questions it could answer.

Ingest-time only, and batched. Query time is untouched -- the entire point is that
the cost is paid once per chunk instead of once per question.

The hard boundary
-----------------
These labels are **a retrieval aid and nothing else**. A wrong auto-tag must not
be able to corrupt an answer, and that is enforced structurally rather than
promised:

* :func:`annotate` writes **only** the four fields this module owns. Whatever else
  a model returns is discarded, so a hallucinated ``heading_path`` or ``page``
  cannot overwrite Layer A through a dict update.
* ``page_content`` is never touched.
* All four fields are in :data:`chunk_schema.FILTER_ONLY`, so none of them reaches
  the prompt. The model that writes the answer never sees a generated label.

Failure is not fatal. A failed or malformed extraction leaves the chunk with the
schema defaults -- still indexed, still retrievable by dense and keyword search,
still citable. Losing a label must never lose a chunk.
"""

from __future__ import annotations

import json
import logging

import backend.config as config
from backend.chunk_schema import CHUNK_SCHEMA, CONTENT_TYPES, LIST_DELIMITER

logger = logging.getLogger("smartdoc.semantic")

# The four fields Layer B owns. Nothing else may be written by this module.
FIELDS: tuple[str, ...] = ("content_type", "topics", "entities", "answers_questions")

_PROMPT = """You label passages from company documents so a search index can find \
them. Reply with JSON only.

For EACH numbered passage return an object with:
  "id": the passage number
  "content_type": exactly one of policy | procedure | table | definition | other
  "topics": 2-5 short topic phrases (lowercase, no sentences)
  "entities": named things it mentions -- roles, teams, systems, codes, document \
names. [] if none.
  "answers_questions": 2-3 natural questions THIS passage answers, phrased as an \
employee would ask them

Rules:
- Describe only what the passage says. Add no facts and infer no context beyond it.
- Do NOT copy figures, amounts or dates into any field. These fields exist to \
FIND the passage, not to state its values.
- content_type: "procedure" for ordered steps, "table" for tabular rows, \
"definition" for a term being defined, "policy" for a rule or entitlement, \
otherwise "other".
- Return one object per passage, in order, as {"passages": [...]}."""


def defaults() -> dict:
    """The schema defaults for Layer B's fields -- what a chunk gets with no labels."""
    return {field: CHUNK_SCHEMA[field] for field in FIELDS}


def _clean_list(value, limit: int) -> str:
    """Normalise a model-returned list into a delimited scalar string."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ""
    out: list[str] = []
    for entry in value:
        text = " ".join(str(entry).split()).strip(" .;,")
        if text and text.lower() not in {o.lower() for o in out}:
            out.append(text[:120])
        if len(out) >= limit:
            break
    return LIST_DELIMITER.join(out)


def _clean_content_type(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in CONTENT_TYPES else CHUNK_SCHEMA["content_type"]


def _batch_prompt(texts: list[str]) -> str:
    limit = config.SEMANTIC_METADATA_CHARS
    return "\n\n".join(
        f"--- passage {index} ---\n{text[:limit]}" for index, text in enumerate(texts, 1)
    )


def extract_batch(texts: list[str], model: str | None = None) -> list[dict]:
    """Label one batch of chunk texts. One field dict per input, in order.

    Never raises: any failure yields schema defaults for the whole batch, which is
    indistinguishable from the flag being off.
    """
    if not texts:
        return []

    try:
        import openai

        from backend.vectorstore import _shared_openai
    except Exception:  # pragma: no cover - import-time environment problem
        return [defaults() for _ in texts]

    try:
        completion = _shared_openai().chat.completions.create(
            model=model or config.UTILITY_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": _batch_prompt(texts)},
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except (openai.OpenAIError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Semantic extraction failed for %d chunk(s): %s", len(texts), exc)
        return [defaults() for _ in texts]

    passages = payload.get("passages")
    if not isinstance(passages, list):
        return [defaults() for _ in texts]

    # Indexed by the model's own id, not by position. A batch returned short or out
    # of order would otherwise shift every label by one and attach a chunk's topics
    # to its neighbour -- silently, since both are plausible strings.
    by_id: dict[int, dict] = {}
    for position, entry in enumerate(passages, start=1):
        if not isinstance(entry, dict):
            continue
        try:
            key = int(entry.get("id", position))
        except (TypeError, ValueError):
            key = position
        by_id.setdefault(key, entry)

    out: list[dict] = []
    for position in range(1, len(texts) + 1):
        entry = by_id.get(position)
        if not isinstance(entry, dict):
            out.append(defaults())
            continue
        out.append(
            {
                "content_type": _clean_content_type(entry.get("content_type")),
                "topics": _clean_list(entry.get("topics"), 5),
                "entities": _clean_list(entry.get("entities"), 8),
                "answers_questions": _clean_list(entry.get("answers_questions"), 3),
            }
        )
    return out


def annotate(documents: list, batch_size: int | None = None) -> int:
    """Attach Layer B fields to every chunk in ``documents``, in place.

    Returns how many chunks came back with at least one populated field. With the
    flag off the schema defaults are written rather than nothing, so a chunk's
    metadata SHAPE never depends on the flag -- only its values do.

    A table part or summary is skipped by the model and typed directly: its
    ``content_type`` is known structurally, and asking an LLM to confirm what the
    table extractor already proved would be paying for a worse answer.
    """
    if not documents:
        return 0

    for document in documents:
        document.metadata.setdefault("content_type", CHUNK_SCHEMA["content_type"])

    tables = [d for d in documents if d.metadata.get("table_id")]
    for document in tables:
        document.metadata.update(defaults())
        document.metadata["content_type"] = "table"

    targets = [d for d in documents if not d.metadata.get("table_id")]
    if not config.SEMANTIC_METADATA_ENABLED:
        for document in targets:
            document.metadata.update(defaults())
        return 0

    size = batch_size or config.SEMANTIC_METADATA_BATCH_SIZE
    annotated = 0
    for start in range(0, len(targets), size):
        batch = targets[start : start + size]
        results = extract_batch([d.page_content for d in batch])
        for document, fields in zip(batch, results):
            # ONLY the four owned fields are written, whatever the extractor
            # returned. This is the hard constraint made structural: a model that
            # emitted "heading_path" or "page" would otherwise overwrite Layer A
            # through update(), and a fabricated heading path would then be both
            # filterable and citable.
            safe = {key: fields.get(key, CHUNK_SCHEMA[key]) for key in FIELDS}
            safe["content_type"] = _clean_content_type(safe["content_type"])
            document.metadata.update(safe)
            if any(safe[f] for f in ("topics", "entities", "answers_questions")):
                annotated += 1
    return annotated


def keyword_side_channel(metadata: dict) -> str:
    """Layer B text added to the KEYWORD index only.

    The single place a generated label is allowed to influence retrieval. Appended
    to the BM25 token stream, never to ``page_content`` -- so a chunk can be
    *found* by a question phrased in words it does not contain, while the prompt
    still receives only what the document actually says.
    """
    parts = [
        str(metadata.get("topics") or ""),
        str(metadata.get("entities") or ""),
        str(metadata.get("answers_questions") or ""),
    ]
    return " ".join(part for part in parts if part)
