"""ColPali query path: encode question -> MaxSim retrieval over page images ->
visual sibling expansion -> vision-grounded generation.

A parallel module to ``backend.rag`` -- not a modification of it (the brief's
hard constraint).

**Expansion gate.** The brief asks for this to key off query intent
(aggregation/enumeration/exhaustive vs. simple lookup). The obvious reuse
candidate is ``backend.query_analysis.analyze``, the hybrid pipeline's own
classifier -- but it was tried and rejected for two independent reasons,
both measured, not assumed:

1. Its actual gate for the analogous hybrid-pipeline feature
   (``retrieval.expand_table_siblings``) does NOT key off query intent at all
   -- it expands unconditionally whenever ANY retrieved unit's metadata shows
   table membership ("a metadata fetch, not a search" -- retrieval.py's own
   docstring). That works there because the hybrid pipeline's retrieval unit
   is a small chunk/table-row-part: "expand to the whole table" still means
   fetching just that table's rows.
2. ``analyze()`` is heuristic-first (``_looks_simple`` short-circuits before
   any LLM call runs), tuned for choosing ``final_k``/``candidate_k`` breadth,
   and its ``_EXHAUSTIVE_RE`` requires "all/every/list/how many
   (different|types|kinds)" -- a plain "How many employees are on leave?"
   matches none of that and is classified ``fact_lookup``, so it would never
   trigger expansion for exactly the aggregation case the brief's acceptance
   check exercises.

Unlike the hybrid pipeline, this module's retrieval unit is a WHOLE PAGE
IMAGE. When an entire multi-page document is one visual table group (as
``Large_Multi_Page_Tables_Test.pdf`` is), a metadata-only gate sends every
page to the vision model even for "What department is Employee 1 in?" --
exactly the over-expansion the brief's acceptance check says must not happen.
Metadata-only gating was tried and measured to fail that check.

So this module requires BOTH signals, closest to the brief's literal
paragraph: a page must belong to a visual table group (metadata, from Phase
1's clustering) AND the question itself must read as
aggregation/enumeration/exhaustive intent. Intent here is a small local
regex (:func:`_is_exhaustive_intent`), deliberately not
``query_analysis.heuristic_type`` -- it needs to also catch bare counting
questions ("how many <plural noun>") that ``_EXHAUSTIVE_RE`` misses, so it is
this module's own narrow concern rather than a second, drifting copy of the
hybrid classifier's word list. It is still "a property of the question, not
something ColPali needs to solve itself" (the brief's own framing) -- just a
regex tuned for this module's actual failure mode instead of reused verbatim.

The refusal phrasing below is copied from ``backend.rag`` as a literal string
constant (not imported), so a user sees identical wording regardless of which
path answered -- a UX consistency choice, not a code dependency: nothing here
imports ``backend.rag`` or touches its state.

Everything else -- retrieval, expansion, generation, citations -- is new code
against this experiment's own store, per the hard isolation rules.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field

import torch
from PIL import Image

from backend.config import OPENAI_API_KEY
from backend.rag import InvalidQuestionError
from colpali_experiment import store
from colpali_experiment.embedder import embed_queries, score
from colpali_experiment.renderer import render_pdf_pages

logger = logging.getLogger("colpali_experiment.answer")

DEFAULT_TOP_K = 5
GENERATION_MODEL = "gpt-4o-mini"

# Same wording as backend.rag.REFUSAL_MESSAGE, kept as a literal so this
# module has zero import dependency on the hybrid pipeline -- only the user-
# visible phrasing is shared, not any code path.
REFUSAL_MESSAGE = "I don't know based on the available documents."

# Generic answer-shape and partial-coverage rules, ported from the same
# categories of instruction backend.rag._BASE_RULES/_FORMAT_RULES already
# proved out (see backend/rag.py) -- NOT copied as per-question fixes. Both
# rules below are properties of the QUESTION/ANSWER pair in general (does the
# answer compare things? does the context cover only part of it?), so they
# generalize across the whole gold set rather than overfitting to any single
# question. Added after the full gold-set comparison run showed a genuine,
# repeated pattern in two categories:
#   - `comparison`: every failure had fully correct content, just never
#     rendered as a table (4/4 failures, 100% of that category).
#   - `out_of_scope_partial`: the model answered a mixed-scope question in
#     full instead of fencing the unsupported half.
# Both are the SAME root cause -- this prompt had no instruction at all for
# either "how much to answer when context is partial" or "what shape to use
# for a multi-entity comparison" -- so one general rule per issue, not a fix
# per failing question.
_SYSTEM_PROMPT = (
    "You answer questions using ONLY the document page images provided to "
    "you. Use ONLY what is visible in these images. Never use outside "
    "knowledge or assumptions, even when you are confident of the real "
    f'answer. If the pages shown do not answer the question at all, reply '
    f'with EXACTLY this sentence and nothing else: "{REFUSAL_MESSAGE}"\n\n'
    "If the question itself is not a real, answerable question -- random "
    "characters, keyboard mashing, or punctuation with no actual words -- "
    f'reply with EXACTLY this sentence and nothing else: "{REFUSAL_MESSAGE}" '
    "Do not describe or summarize the page images in that case; a page "
    "being visible is not the same as the question being real.\n\n"
    "How much to answer:\n"
    "- If the images fully answer the question, answer it completely.\n"
    "- If the images answer only PART of the question, answer that part, "
    "then add one short final sentence naming exactly what the images do "
    "not cover. Do NOT use the fixed refusal sentence in this case -- that "
    "sentence is only for a question the images do not address at all.\n\n"
    "Structure -- match the shape of the answer to its content:\n"
    "- Use a markdown table ONLY when the question explicitly asks you to "
    "compare, or contrast, two or more named things ACROSS THE SAME SET OF "
    "ATTRIBUTES (for example: compare band X and band Y's leave, notice, "
    "and bonus eligibility side by side). One column per attribute, a "
    "header row, one row per item. Every entity named in the question must "
    "get its own row -- do not drop one because the images cover it less "
    "fully.\n"
    "- Do NOT use a table for a question that only asks what something IS "
    "or asks to list items of one kind (\"what are the bands\", \"list the "
    "non-reimbursable items\") -- answer those as prose or a plain list, "
    "even if you could describe each item with several attributes.\n"
    "- Do NOT use a table for a question asking HOW one thing differs from "
    "another in general (\"how does X differ from Y\") unless it also names "
    "the specific attributes to compare -- answer that as prose describing "
    "the difference, since forcing unrelated facts into matched table rows "
    "can misstate what is actually comparable between them.\n\n"
    "Counting and listing:\n"
    "- If the question asks 'how many' of something, or asks for a list of "
    "items, name every individual item you can see in the images, not just "
    "the count. A number without the items it counts is an incomplete "
    "answer even if the number itself is correct.\n"
    "- Items of the same kind can appear on more than one page, including "
    "the last page shown to you, not only the first -- check every page "
    "before finishing the list."
)

# Aggregation/enumeration/exhaustive intent, as a property of the question
# text -- same spirit as backend.query_analysis._EXHAUSTIVE_RE, but also
# catches bare "how many <plural noun>" counting questions ("how many
# employees are on leave"), which that regex requires a following
# different/types/kinds to match and so misses. See the module docstring for
# why this is a small local regex rather than a reuse of that classifier.
_EXHAUSTIVE_INTENT_RE = re.compile(
    r"\b(all|every|each|list|enumerate|complete list|full list|total number|"
    r"how many)\b",
    re.I,
)


def _is_exhaustive_intent(question: str) -> bool:
    return bool(_EXHAUSTIVE_INTENT_RE.search(question))


@dataclass
class RetrievedPage:
    document_id: str
    filename: str
    page_number: int
    score: float
    source: str = "top_k"  # "top_k" | "sibling_expansion"


@dataclass
class VisualAnswer:
    answer: str
    pages: list[RetrievedPage] = field(default_factory=list)
    expanded: bool = False
    pending_documents: list[str] = field(default_factory=list)
    # Token usage for the vision generation call only (mirrors
    # backend.rag.query's own "generation_usage" diagnostics field) -- lets
    # the eval harness compute a comparable per-query cost. None when there
    # was no generation call at all (refusal/pending/no-pages paths).
    generation_usage: dict | None = None


STILL_INDEXING_MESSAGE = (
    "This document is still indexing for visual search -- try again shortly."
)


def _rank_pages(user_id: str, question: str, top_k: int) -> list[RetrievedPage]:
    """MaxSim-rank this user's stored pages against ``question``.

    Scoped to ``user_id`` at the store layer (``store.get_user_embeddings``
    filters by ``user_id`` in SQL) -- the same per-user isolation discipline
    the hybrid pipeline applies via ``user_scope``, just against this new
    store instead of ``chroma_store``/``smartdoc.db``.
    """
    rows = store.get_user_embeddings(user_id)
    if not rows:
        return []

    query_embeddings = embed_queries([question])
    page_embeddings = [torch.from_numpy(r["embedding"]) for r in rows]
    scores = score(query_embeddings, page_embeddings)[0].tolist()

    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedPage(
            document_id=r["document_id"],
            filename=r["filename"],
            page_number=r["page_number"],
            score=s,
        )
        for r, s in ranked[:top_k]
    ]


def _expand_visual_siblings(user_id: str, pages: list[RetrievedPage]) -> list[RetrievedPage]:
    """For each retrieved page that belongs to a visual table group (Phase 1),
    pull in every other page of that group -- not just whichever subset top-k
    happened to rank highly. Pure metadata lookup against the already-loaded
    ``table_group_id`` column; no re-scoring, no extra model calls.
    """
    rows = store.get_user_embeddings(user_id)
    by_document: dict[str, list[dict]] = {}
    for r in rows:
        by_document.setdefault(r["document_id"], []).append(r)

    seen = {(p.document_id, p.page_number) for p in pages}
    expanded = list(pages)
    for page in pages:
        doc_rows = by_document.get(page.document_id, [])
        this_row = next(
            (r for r in doc_rows if r["page_number"] == page.page_number), None
        )
        group_id = this_row.get("table_group_id") if this_row else None
        if not group_id:
            continue
        for sibling in doc_rows:
            key = (sibling["document_id"], sibling["page_number"])
            if sibling.get("table_group_id") != group_id or key in seen:
                continue
            seen.add(key)
            expanded.append(
                RetrievedPage(
                    document_id=sibling["document_id"],
                    filename=sibling["filename"],
                    page_number=sibling["page_number"],
                    score=0.0,
                    source="sibling_expansion",
                )
            )
    expanded.sort(key=lambda p: (p.document_id, p.page_number))
    return expanded


def _resolve_page_image(page: RetrievedPage, user_id: str) -> Image.Image | None:
    """Reuse the same two-location file lookup the rest of this experiment
    uses, so this module never guesses a path convention of its own.
    """
    from backend import db as backend_db
    from colpali_experiment.ingest import _resolve_pdf_path

    record = backend_db.get_document(user_id, page.document_id)
    if record is None:
        return None
    pdf_path = _resolve_pdf_path(user_id, record["filename"])
    if pdf_path is None:
        return None
    rendered = dict(render_pdf_pages(pdf_path, page.document_id))
    return rendered.get(page.page_number)


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _generate(
    question: str, pages: list[RetrievedPage], images: list[Image.Image]
) -> tuple[str, dict | None]:
    """Returns ``(answer_text, generation_usage)``.

    ``generation_usage`` mirrors ``backend.rag.query``'s own diagnostics field
    (model name + prompt/completion/total tokens) so the eval harness can
    compute cost identically for both backends -- the prompt token count here
    is dominated by IMAGE tokens (each page image costs meaningfully more
    tokens than the equivalent text chunk would), which is exactly the
    "vision-input pricing, different from text-only calls" distinction the
    comparison report has to keep honest rather than normalize away.
    """
    import openai

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured (backend.config.OPENAI_API_KEY).")

    content: list[dict] = [{"type": "text", "text": question}]
    for page, image in zip(pages, images):
        content.append(
            {
                "type": "text",
                "text": f"--- {page.filename}, page {page.page_number} ---",
            }
        )
        content.append(
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}}
        )

    client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    answer_text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    generation_usage = (
        {
            "model": GENERATION_MODEL,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        if usage is not None
        else None
    )
    return answer_text, generation_usage


def answer(user_id: str, question: str, top_k: int = DEFAULT_TOP_K) -> VisualAnswer:
    """Full visual query path: retrieve -> (maybe) expand -> generate.

    ``top_k`` pages are ranked purely by MaxSim first. Expansion to a page's
    full visual table group fires only when BOTH: (a) a top-k page carries a
    ``table_group_id``, and (b) the question itself reads as
    aggregation/enumeration/exhaustive intent (:func:`_is_exhaustive_intent`).
    See the module docstring for why both signals are required here, unlike
    the hybrid pipeline's metadata-only gate -- a whole-page retrieval unit
    makes unconditional expansion of a large table group expensive and wrong
    for a plain single-fact lookup.

    Before returning ANY refusal, checks this user's ColPali ingestion status
    (``colpali_experiment.store.colpali_ingest_status``, written by
    ``/upload``'s background fan-out) and annotates it with
    :data:`STILL_INDEXING_MESSAGE` if the user has a document still
    'pending'. This has to be a check on every refusal path, not just "no
    pages matched at all": a user can have some documents ready and others
    still indexing, and a plain per-user MaxSim scan will happily rank pages
    from the ready ones and let the model refuse based on those alone -- a
    refusal in that situation reads as "the documents don't contain this",
    which is a different, misleading claim from "the specific document that
    might have answered this hasn't finished indexing for visual search
    yet". Measured, not assumed: this exact gap was found by testing a
    question against a just-uploaded (still-pending) document while another
    of the same user's documents was already ready -- the bare "no ranked
    pages" check missed it because pages from the ready document DID rank.
    """
    question = question.strip()
    if not question:
        # Same guard and same exception type as backend.rag.query (see
        # InvalidQuestionError there) -- main.py's global exception handler
        # already turns this into a 400, and reusing it means a blank
        # question is rejected the same way regardless of which backend
        # answers, rather than reaching retrieval at all. Without this,
        # top_k pages still MaxSim-rank against an empty query string (some
        # page always scores highest) and the model would summarize
        # whatever came back instead of recognizing there was no question.
        raise InvalidQuestionError("Question must not be empty or whitespace-only.")

    def _pending_filenames() -> list[str]:
        statuses = store.get_ingest_statuses_for_user(user_id)
        return sorted({s["filename"] for s in statuses if s["status"] == "pending"})

    ranked = _rank_pages(user_id, question, top_k)
    if not ranked:
        pending = _pending_filenames()
        if pending:
            return VisualAnswer(answer=STILL_INDEXING_MESSAGE, pending_documents=pending)
        return VisualAnswer(answer=REFUSAL_MESSAGE, pages=[])

    expanded = False
    pages = ranked
    if _is_exhaustive_intent(question):
        expanded_pages = _expand_visual_siblings(user_id, ranked)
        if len(expanded_pages) > len(ranked):
            expanded = True
            pages = expanded_pages

    logger.info(
        "colpali answer: question=%r expanded=%s pages=%s",
        question,
        expanded,
        [(p.filename, p.page_number, p.source) for p in pages],
    )

    images = []
    resolved_pages = []
    for page in pages:
        image = _resolve_page_image(page, user_id)
        if image is None:
            continue
        images.append(image)
        resolved_pages.append(page)

    if not images:
        pending = _pending_filenames()
        if pending:
            return VisualAnswer(answer=STILL_INDEXING_MESSAGE, pending_documents=pending)
        return VisualAnswer(answer=REFUSAL_MESSAGE, pages=[], expanded=expanded)

    generated, generation_usage = _generate(question, resolved_pages, images)
    if generated.strip() == REFUSAL_MESSAGE:
        # The model itself refused based on the pages it was shown -- still
        # ambiguous if this user has anything pending, for the same reason as
        # the two checks above: the retrieved pages weren't the answer, but a
        # pending document might have been.
        pending = _pending_filenames()
        if pending:
            return VisualAnswer(
                answer=STILL_INDEXING_MESSAGE,
                pages=resolved_pages,
                expanded=expanded,
                pending_documents=pending,
            )
    return VisualAnswer(
        answer=generated,
        pages=resolved_pages,
        expanded=expanded,
        generation_usage=generation_usage,
    )


def to_rag_response(visual_answer: VisualAnswer):
    """Adapt a :class:`VisualAnswer` into a ``backend.rag.RagResponse``.

    This is the ONLY point of contact this module has with ``backend.rag``:
    it imports two plain dataclasses (``RagResponse``, ``Source``) purely for
    their SHAPE, never the ``query()`` function or any hybrid retrieval/
    generation logic. ``/ask`` in ``backend/main.py`` persists a turn (session
    messages, ``last_document``, background summarization) using only
    ``response.answer``, ``response.sources[0].source``, and
    ``response.to_dict()`` -- all of which a ``RagResponse`` built from
    ColPali's own output satisfies identically, so that persistence code runs
    completely unmodified regardless of which backend answered.

    A page-level ColPali citation has no fine-grained ``section``/heading
    (there was no text extraction to derive one from), so ``section=""`` --
    ``SourcesPanel`` on the frontend already renders that fine, omitting the
    section line entirely (see ``web/src/components/chat/sources-panel.tsx``).
    ``snippet`` is a synthesized caption rather than a quoted excerpt, since
    the model read a whole page image, not a text span.
    """
    from backend.rag import RagResponse, Source

    sources = [
        Source(
            source=page.filename,
            page=page.page_number,
            snippet=(
                f"Page {page.page_number} of {page.filename} (visual/page-image "
                f"citation -- no text snippet, the model read the page image)."
            ),
            section="",
        )
        for page in visual_answer.pages
    ]
    # "expanded" surfaces whether _expand_visual_siblings actually widened the
    # page set for this answer -- the only place that signal is otherwise
    # visible is this module's own return value, and the eval harness (whose
    # only view of the system is the /ask HTTP boundary, by design -- see
    # eval/eval_tool/api_client.py) needs it to attribute a category-level
    # pass-rate gap to retrieval architecture rather than partly to whichever
    # backend's intent classifier happened to fire.
    diagnostics: dict = {"expanded": visual_answer.expanded}
    if visual_answer.generation_usage is not None:
        diagnostics["generation_usage"] = visual_answer.generation_usage
    return RagResponse(
        answer=visual_answer.answer,
        sources=sources,
        query_type="colpali",
        diagnostics=diagnostics,
    )
