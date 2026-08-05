"""The core RAG query function: plan, retrieve, assemble, answer, verify.

    analyze -> route -> retrieve -> assemble -> generate -> ground-check -> cite

Design commitments
------------------
**Citations cannot diverge from context.** Sources are built from
``AssembledContext.units_used`` -- exactly the units that entered the prompt. A
previous implementation passed *all* retrieved hits to the prompt while filtering
the citation list by a distance margin, so the model could read four passages
while the user was shown one; any claim drawn from the other three was uncited.

**Citations cannot be hallucinated.** Every field of every source comes from
retrieval metadata and chunk text. Nothing parses the model's output.

**Snippets are language-independent.** The snippet is cut from the child chunk
that actually matched the query embedding, not chosen by lexical overlap with the
question -- which silently degrades to head-of-section for any non-English
question (observed: a Spanish query about annual leave citing the Sick Leave
section).

**Refusal is graded, not binary.** A half-answerable question used to produce an
answer with the refusal sentence bolted on. Three cases are distinguished, and
only "nothing answerable" emits the fixed refusal string.

**Answers are checked AND enforced.** Detecting an unsupported claim and
returning it anyway defeats the purpose of detecting it: verification failures
are regenerated, pruned, or withdrawn -- subject to guards that stop remediation
from doing net harm.

**Voice and structure are prompt-side, and cannot move a value.** Phase 4 asks
for a warm human register and for an answer's shape to match its content -- a
table for a comparison, a list for steps, prose for an explanation. Both are
instructions to the answer model (``_VOICE_RULES``, ``_FORMAT_RULES``); no code
here parses, rewrites, or re-flows the model's output, so formatting cannot
change a figure or a citation. The two places where it *could* have changed an
answer's meaning are handled explicitly: refusal detection ignores markdown
decoration (``_normalise_answer``), and grounding remediation prunes whole
block lines rather than sentences (``_split_units``), so a table row is dropped
as a row instead of being flattened into prose.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import openai

import backend.config as config
from backend.context import AssembledContext, assemble
from backend.ingestion import count_tokens
from backend.query_analysis import (
    COMPARISON,
    CROSS_DOCUMENT,
    EXHAUSTIVE,
    MULTI_HOP,
    PROCEDURAL,
    PROFILES,
    SYNTHESIS,
    QueryPlan,
    analyze,
)
from backend.intent import COMBINATION, DOCUMENT_WIDE
from backend.intent import classify as intent_of
from backend.retrieval import RetrievalResult, retrieve
from backend.user_scope import current_user_id
from backend.vectorstore import VectorStoreError, _shared_openai  # noqa: F401

logger = logging.getLogger("smartdoc.rag")

REFUSAL_MESSAGE = "I don't know based on the available documents."

SNIPPET_LENGTH = 260

# Addendum 2. A small, long-lived pool that runs the speculative SQL lookup
# CONCURRENTLY with the hybrid pipeline, so total latency is max(the two) rather
# than their sum.
#
# Threads rather than asyncio: ``query`` and ``retrieve`` are synchronous and are
# called from sync FastAPI handlers, scripts, and tests, so making this branch
# async would mean colouring the whole call graph to save nothing -- the SQL side
# is one blocking sqlite3 call, which releases the GIL for its duration, and the
# vector side is blocking HTTP. Two threads is the shortest path to real overlap.
#
# Two workers, not one: a single worker would serialise two concurrent requests'
# lookups behind each other, and not many, because each task is a ~1ms indexed
# read and a deep pool would only add idle threads.
_SQL_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="smartdoc-sql")

# Ceiling on how long the answer path will wait for a lookup that should take a
# millisecond. If SQLite is locked behind a long write, the passages answer alone
# -- which is the flag-off behaviour, so the timeout degrades rather than fails.
_SQL_TIMEOUT_SECONDS = 2.0

# Numbers and identifier-shaped tokens are the claims most worth verifying: a
# wrong entitlement figure or fault code is the failure mode that matters in
# policy and manual Q&A.
_NUMERIC_CLAIM_RE = re.compile(r"\b\d[\d,.]*\b")
_IDENT_CLAIM_RE = re.compile(r"\b[A-Z]{1,4}-\d{1,4}\b|\bAES-?\d{3}\b|\bTLS\s?\d\.\d\b")


class RagError(Exception):
    """Base class for retrieval/generation failures."""


class InvalidQuestionError(RagError):
    """Raised when the question is empty or whitespace-only."""


class GenerationError(RagError):
    """Raised when the OpenAI chat completion call fails."""


@dataclass
class Source:
    """A structural citation, derived only from retrieval metadata and text."""

    source: str
    page: int
    snippet: str
    section: str = ""
    page_end: int | None = None

    # V3.1: the full heading hierarchy ("Leave Policy > Sick Leave > Eligibility")
    # when the document was ingested as markdown, "" otherwise. Additive -- the
    # existing ``section`` field is unchanged, so a client that ignores this key
    # renders exactly what it rendered before.
    heading_path: str = ""

    # V3.3 Use 3: the citation as a reader should see it --
    # "Leave Policy > Casual Leave > Eligibility, p. 7". Composed server-side so
    # every surface (API, CLI, web) shows the same label, and with a fallback so it
    # is populated on the plain-text path too. Display only, never a source of truth.
    display: str = ""


@dataclass
class Grounding:
    """Verification verdict for an answer.

    Two independent signals, kept separate on purpose:

    * ``unsupported_claims`` -- what the LLM entailment judge rejected.
    * ``unverified_numbers`` -- figures in the answer not verbatim in context.
      Informational: a legitimately derived value ("a difference of eight days"
      from 20 and 28) lands here, so it must not by itself condemn an answer.
    """

    checked: bool = False
    faithful: bool | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    note: str = ""

    # Remediation trail: "regenerated", "pruned", "declined", or "".
    repaired: str = ""
    removed_claims: list[str] = field(default_factory=list)


@dataclass
class RagResponse:
    """Return type of :func:`query`; serialises cleanly to JSON."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    query_type: str = ""
    grounding: Grounding = field(default_factory=Grounding)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [asdict(s) for s in self.sources],
            "query_type": self.query_type,
            "grounding": asdict(self.grounding),
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_BASE_RULES = f"""You are SmartDoc, a company document assistant. You answer \
employee questions using ONLY the context passages provided, each labelled with \
its source document and page.

Grounding rules -- follow all of them:
- Use ONLY information present in the context. Never use outside knowledge or \
assumptions, even when you are confident of the real answer.
- Never invent numbers, dates, names, codes, or thresholds. If a figure is not \
in the context, do not state a figure.
- Do not fabricate citations, page numbers, or filenames in your answer text; \
citations are attached separately by the system.
- Answer naturally, without phrases like "the context says".

Conversation memory:
- You may be given a short summary of this conversation's earlier turns. Use \
it ONLY to resolve references in the question (pronouns, "that policy", "the \
same band") -- never as a source of facts. Every factual claim must still come \
from the context passages below, even if the summary already states it.

How much to answer:
- If the context fully answers the question, answer it completely.
- If the context answers PART of the question, answer that part, then add one \
short final sentence naming exactly what the documents do not cover. Do NOT use \
the fixed refusal sentence in this case.
- If the context does not answer the question at all, reply with EXACTLY this \
sentence and nothing else: "{REFUSAL_MESSAGE}"
- If two documents give conflicting values, report both and say which document \
each came from. Do not silently pick one."""

_TYPE_RULES = {
    COMPARISON: """This is a COMPARISON question. Address every entity named in \
the question explicitly, even if the context covers one better than another. \
State each side's value, then the difference. If the context covers only one \
side, say which side is missing rather than implying symmetry.""",
    MULTI_HOP: """This is a MULTI-STEP question. The answer depends on an \
intermediate fact (for example an entity's category, tier, or classification). \
Identify that intermediate fact from the context first, then apply the rule that \
governs it, and state both links explicitly so the reasoning is checkable.""",
    PROCEDURAL: """This is a PROCEDURAL question. Answer as an ordered list of \
steps in the order the documents give them. Preserve every stated deadline, \
threshold, and approver. Do not merge or reorder steps, and do not invent steps \
to bridge a gap in the context -- note the gap instead.""",
    SYNTHESIS: """This is a DOCUMENT-WIDE SYNTHESIS question. The context is \
presented in document order and may begin with the document's section outline.

- Organise your answer to follow the document's own structure, using short \
headings drawn from its sections.
- Cover every section present in the context rather than generalising across \
them; a synthesis that collapses five sections into one paragraph has lost the \
information the question asked for.
- Preserve concrete values (amounts, deadlines, thresholds) where the context \
gives them.
- If the outline lists sections the context does not include, say which parts of \
the document your answer does not cover. Do not imply the excerpt is the whole \
document.""",
    EXHAUSTIVE: """This is an EXHAUSTIVE EXTRACTION question. Completeness is the \
priority. Enumerate EVERY matching item found anywhere in the context, including \
items inside tables AND items described only in prose outside tables. Do not \
summarise, do not truncate with "etc.", and do not stop at the first list you \
find -- items of the same kind may appear in more than one passage. If you \
believe the list may be incomplete, say so explicitly at the end.""",
    CROSS_DOCUMENT: """This is a CROSS-DOCUMENT question. Several documents are \
in the context by design.

- Attribute every value to the document it came from, by document name only. \
Never write page numbers in your answer text -- the system attaches exact pages \
separately, and a page number you invent cannot be checked.
- Where documents state DIFFERENT values for the same thing, present that \
explicitly as a difference (or a conflict) rather than choosing one.
- Where documents agree, say so once rather than repeating it per document.
- If one document covers the subject and another is silent, say the second is \
silent rather than treating silence as agreement.""",
}

# Phase 4, Part A. Tone. Gated by ANSWER_VOICE_ENABLED.
#
# The register is the only thing being asked for here, and the last two bullets
# are the load-bearing ones: a friendly instruction is exactly the kind of thing
# that invites a model to add a reassurance the documents do not support, or to
# wrap the fixed refusal sentence in an apology -- which would break
# ``_is_refusal`` and hand citations to an answer that refused.
_VOICE_RULES = f"""Voice -- how the answer should read:
- Write like a helpful colleague explaining something to a co-worker: warm, \
plain, and direct. Contractions are welcome ("you'll", "it's", "there's").
- Say "you" when the answer is about the reader. Lead with the answer itself, \
never with a preamble.
- No corporate register ("please be advised", "the aforementioned"), no filler \
openers ("Great question!", "Certainly!"), no sign-offs, and no offers to help \
further.
- Warmth is a matter of WORDING ONLY. Never add a fact, a reassurance, a \
recommendation, or a caveat the context does not state, and never soften or \
hedge a value the context gives plainly.
- One exception, absolute: when the context does not answer the question at \
all, return the refusal sentence EXACTLY as written above -- \
"{REFUSAL_MESSAGE}" -- with no apology before it, no offer after it, and no \
bold, quotes, or bullet around it."""

# Phase 4, Part A. Structure. Gated by ANSWER_FORMAT_ENABLED.
#
# The model chooses the shape; nothing downstream imposes one. The second block
# exists because a tidy table is a strong pull toward rounding a value,
# inventing a missing cell, or reordering rows -- each of which would be
# formatting altering correctness, which this phase forbids outright.
_FORMAT_RULES = """Structure -- match the shape of the answer to its content:
- TABLE: use a markdown table when the answer compares two or more things, or \
lists items that each carry the same two or more attributes (an entity and its \
tier, a fault code and its meaning, a band and its entitlement). One column per \
attribute, a header row, one row per item.
- BULLETS: use a bulleted list for a set of separate points, and a NUMBERED \
list when order matters (steps, a sequence, a procedure). One point per item; \
do not nest more than one level.
- PROSE: use short paragraphs for an explanation, a definition, a single fact, \
or a statement about what the documents do and do not cover. A one-sentence \
answer is a sentence, not a one-item list.
- Add a short bold lead-in or a short heading only when the answer has more \
than one distinct part. Do not decorate a short answer.
- Where an instruction above already names a structure for this question type, \
that instruction wins; these rules only decide the shape when it does not.

Structure rules that protect correctness -- these override every preference \
above:
- Formatting NEVER changes a value. Copy each number, date, code, and name \
exactly as the context gives it. Do not round, convert units, re-order, or \
normalise anything to make a table tidy.
- If the context gives no value for a cell, write "Not specified". Never fill a \
gap to complete a row.
- Never put page numbers or filename citations in a table or a list; the system \
attaches exact sources separately.
- If the content will not fit a table honestly (one side is missing, the values \
are not comparable), say that in prose rather than forcing the table.
- The refusal sentence, and the one short sentence naming what the documents do \
not cover, are always plain prose."""


def _system_prompt(plan: QueryPlan) -> str:
    """Base grounding rules, this query type's instructions, then voice/format.

    Voice and structure come LAST so the grounding and type rules are what the
    model reads first, and so each Phase 4 block is a clean textual addition --
    with both flags off, this returns byte-for-byte what Phase 3 returned.
    """
    parts = [_BASE_RULES, _TYPE_RULES.get(plan.query_type, "")]
    if config.ANSWER_VOICE_ENABLED:
        parts.append(_VOICE_RULES)
    if config.ANSWER_FORMAT_ENABLED:
        parts.append(_FORMAT_RULES)
    return "\n\n".join(p for p in parts if p).strip()


def build_prompt(
    question: str,
    context: AssembledContext,
    conversation_context: str | None = None,
    exact_facts: str = "",
) -> str:
    """Assemble the user turn: conversation memory, exact facts, context, question.

    ``conversation_context`` (a per-session running summary) is placed BEFORE
    the retrieved context and labelled distinctly from it, so the system
    prompt's "memory resolves references, never supplies facts" rule has a
    visibly separate block to point at rather than an ambiguous blend.

    ``exact_facts`` (Addendum 2) is the confidently-resolved table cell, rendered
    by ``table_store.render_facts``. It sits ABOVE the passages, labelled
    authoritative, so the model knows which of the two wins for a value while
    still having the passages for everything around it. Empty string -- the
    normal case -- reproduces the previous prompt byte for byte.
    """
    body = context.text or "(no relevant context retrieved)"
    memory_block = (
        f"Conversation memory (for resolving references only):\n{conversation_context}\n\n"
        if conversation_context
        else ""
    )
    facts_block = f"{exact_facts}\n\n" if exact_facts else ""
    closing = (
        "Answer using only the exact facts and the context above."
        if exact_facts
        else "Answer using only the context above."
    )
    return (
        f"{memory_block}"
        f"{facts_block}"
        f"Context passages:\n\n{body}\n\n"
        f"Question: {question}\n\n"
        f"{closing}"
    )


# ---------------------------------------------------------------------------
# Refusal / incompleteness detection
# ---------------------------------------------------------------------------

_QUOTES = "\"'`“”‘’"

# The base prompt instructs a partial answer to end with one sentence naming what
# the documents do not cover. Because we control that phrasing, matching it is a
# reliable signal that retrieval came back incomplete -- worth one wider retry,
# exactly like a refusal.
_INCOMPLETE_RE = re.compile(
    r"\b(do(?:es)?\s+not\s+(?:cover|specify|state|mention|provide|include)|"
    r"not\s+specified|no\s+information\s+(?:about|on)|documents?\s+do\s+not)\b",
    re.I,
)


# Markdown that can wrap or lead a line without changing a single word of it.
# Stripped before refusal comparison: once the model is told to format answers,
# a refusal comes back as "**I don't know based on the available documents.**"
# or as a lone bullet often enough to matter, and an undetected refusal is the
# expensive kind of miss -- `query()` would skip escalation and then attach
# citations to an answer that claims nothing.
_MD_LEAD_RE = re.compile(r"^(?:\s*(?:>|#{1,6}|[-*+]|\d+[.)])\s+)+")
_MD_WRAPPERS = ("***", "**", "*", "___", "__", "_", "`")


def _normalise_answer(text: str) -> str:
    """Normalise for refusal comparison: whitespace, markdown, quotes, case.

    Only DECORATION is removed -- markers and quotes, never words -- so this
    cannot turn a substantive answer into the refusal string: the remaining
    text still has to match the refusal sentence word for word.
    """
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    changed = True
    while changed and collapsed:
        changed = False
        lead_stripped = _MD_LEAD_RE.sub("", collapsed).strip()
        if lead_stripped != collapsed:
            collapsed, changed = lead_stripped, True
        for wrapper in _MD_WRAPPERS:
            if (
                len(collapsed) > 2 * len(wrapper)
                and collapsed.startswith(wrapper)
                and collapsed.endswith(wrapper)
            ):
                collapsed = collapsed[len(wrapper) : -len(wrapper)].strip()
                changed = True
                break
        if len(collapsed) > 1 and collapsed[0] in _QUOTES and collapsed[-1] in _QUOTES:
            collapsed, changed = collapsed[1:-1].strip(), True
    return collapsed.casefold()


def _is_refusal(answer_text: str) -> bool:
    """True if the WHOLE answer is the refusal sentence.

    Compares the entire normalised answer, not a substring: a genuine partial
    answer may legitimately contain the same sentence as one clause among others,
    and that case must keep its citations.
    """
    return _normalise_answer(answer_text) == _normalise_answer(REFUSAL_MESSAGE)


def _signals_incomplete(answer_text: str) -> bool:
    """True if the answer itself reports missing coverage."""
    return bool(_INCOMPLETE_RE.search(answer_text))


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def _strip_breadcrumb(text: str) -> str:
    """Remove the ingestion breadcrumb prefix from a chunk for display."""
    parts = text.split("\n\n", 1)
    if len(parts) == 2 and " > " in parts[0] and len(parts[0]) < 160:
        return parts[1]
    return text


def _snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    """Trim to ``length`` characters on a word boundary."""
    clean = re.sub(r"\s+", " ", _strip_breadcrumb(text)).strip()
    if len(clean) <= length:
        return clean
    cut = clean[:length]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "..."


def citation_display(
    heading_path: str, doc_title: str, section: str, page: int, page_end: int | None
) -> str:
    """Compose the human-readable citation label.

    Prefers the heading path; falls back to "<doc title> > <section>" so a citation
    always carries section context, whichever ingestion path produced the chunk.
    """
    trail = (heading_path or "").strip()
    if not trail:
        trail = " > ".join(
            p for p in ((doc_title or "").strip(), (section or "").strip()) if p
        )
    pages = f"p. {page}" if not page_end or page_end <= page else f"pp. {page}-{page_end}"
    return f"{trail}, {pages}" if trail else pages


def build_sources(context: AssembledContext) -> list[Source]:
    """Build citations from exactly the units that entered the prompt."""
    sources: list[Source] = []
    seen: set[tuple[str, int, str]] = set()
    for unit in context.units_used:
        section = unit.metadata.get("section", "") or ""
        key = (unit.source, unit.page, section)
        if key in seen:
            continue
        seen.add(key)
        page_end_raw = unit.metadata.get("page_end")
        page_end = int(page_end_raw) if page_end_raw else unit.page
        sources.append(
            Source(
                source=unit.source,
                page=unit.page,
                snippet=_snippet(unit.matched_text or unit.text),
                section=section,
                page_end=page_end if page_end > unit.page else None,
                heading_path=str(unit.metadata.get("heading_path", "") or ""),
                display=citation_display(
                    str(unit.metadata.get("heading_path", "") or ""),
                    str(unit.metadata.get("doc_title", "") or ""),
                    section,
                    unit.page,
                    page_end if page_end > unit.page else None,
                ),
            )
        )
    return sources


def add_sql_sources(sources: list[Source], facts: list) -> list[Source]:
    """Append a citation for each confident SQL fact, deduped against passages.

    A SQL-path answer without a citation is a failure -- the reader gets an exact
    number with nothing to check it against, which is strictly worse than the
    hedged answer they would otherwise have had. The cell carries its own
    filename, page, and table title, so no second lookup is needed.

    Deduped by (source, page, table title) so a fact whose table was ALSO
    retrieved as a passage does not produce two entries for one place.
    """
    if not facts:
        return sources
    existing = {(s.source, s.page, s.section) for s in sources}
    out = list(sources)
    for fact in facts:
        key = (fact.source, fact.page, fact.table_title)
        if key in existing:
            continue
        existing.add(key)
        trail = " > ".join(p for p in (fact.table_title, fact.column) if p)
        out.append(
            Source(
                source=fact.source,
                page=fact.page,
                # The snippet is the cell verbatim, not a rendering of it: this
                # is the one citation whose supporting text is a single value.
                snippet=f"{fact.entity} - {fact.column}: {fact.value}",
                section=fact.table_title,
                heading_path=trail,
                display=citation_display(trail, "", fact.table_title, fact.page, None),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Grounding verification
# ---------------------------------------------------------------------------


def _structural_claim_check(answer: str, context_text: str) -> list[str]:
    """Return numeric/identifier claims in ``answer`` absent from context.

    A cheap, deterministic detector for the error class that matters most here: a
    fabricated entitlement figure or fault code.
    """
    # Strip list numbering ("1.", "2)") first: an enumerated answer is
    # formatting, not assertion, and counting its markers as unsupported figures
    # buries real hallucinations in noise. The optional bullet/heading prefix
    # covers the markdown a formatted answer actually produces -- "- 1. ..." and
    # "### 2. ..." both carry a marker the old pattern missed because it required
    # the digit at the start of the line.
    answer = re.sub(r"(?m)^\s*(?:[-*+]\s+|#{1,6}\s+)?\d+[.)]\s+", "", answer)

    context_numbers = {
        n.replace(",", "").rstrip(".") for n in _NUMERIC_CLAIM_RE.findall(context_text)
    }
    context_idents = {
        i.upper().replace(" ", "") for i in _IDENT_CLAIM_RE.findall(context_text)
    }

    unsupported: list[str] = []
    for raw in _NUMERIC_CLAIM_RE.findall(answer):
        value = raw.replace(",", "").rstrip(".")
        if value and value not in context_numbers:
            unsupported.append(raw)
    for raw in _IDENT_CLAIM_RE.findall(answer):
        if raw.upper().replace(" ", "") not in context_idents:
            unsupported.append(raw)
    return list(dict.fromkeys(unsupported))


_FAITHFULNESS_PROMPT = """You verify whether an answer is fully supported by the \
provided context. Reply with JSON only.

An answer is faithful if EVERY factual claim in it is stated in, or directly \
entailed by, the context. It is unfaithful if any claim adds information the \
context does not contain.

Arithmetic the reader could do from stated values IS entailed: if the context \
gives 20 days and 28 days, then "a difference of eight days" is faithful. Do not \
flag such derived values.

Statements about what the context does NOT contain ("the documents do not \
specify X") are not factual claims about the world and must never be flagged.

Ignore style. Judge only support.

Reply exactly: {"faithful": true|false, "unsupported": ["<claim>", ...]}"""


def _llm_faithfulness(answer: str, context_text: str) -> tuple[bool | None, list[str]]:
    """Judge entailment of ``answer`` by ``context_text`` via one LLM call."""
    try:
        completion = _shared_openai().chat.completions.create(
            model=config.UTILITY_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _FAITHFULNESS_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nAnswer:\n{answer}",
                },
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except (openai.OpenAIError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Faithfulness check failed: %s", exc)
        return None, []

    faithful = payload.get("faithful")
    unsupported = [str(c) for c in (payload.get("unsupported") or [])][:8]
    return (bool(faithful) if faithful is not None else None), unsupported


def verify_grounding(
    answer: str, context_text: str, use_llm: bool | None = None
) -> Grounding:
    """Verify ``answer`` against ``context_text`` structurally and by LLM."""
    unverified_numbers = _structural_claim_check(answer, context_text)
    use_llm = config.ENABLE_GROUNDING_CHECK if use_llm is None else use_llm

    faithful: bool | None = None
    unsupported: list[str] = []
    note = ""

    if use_llm:
        faithful, unsupported = _llm_faithfulness(answer, context_text)
        if faithful is None:
            note = "LLM faithfulness check unavailable; structural check only."
            faithful = not unverified_numbers
    else:
        note = "Structural check only (LLM grounding check disabled)."
        faithful = not unverified_numbers

    return Grounding(
        checked=True,
        faithful=faithful,
        unsupported_claims=unsupported,
        unverified_numbers=unverified_numbers,
        note=note,
    )


# ---------------------------------------------------------------------------
# Grounding remediation
# ---------------------------------------------------------------------------

_REPAIR_PROMPT = """You are correcting an answer that contains claims the source \
context does not support.

Rewrite the answer so every remaining statement is fully supported by the \
context. Specifically:
- DELETE each unsupported claim listed below. Do not rephrase or soften it.
- Keep every supported statement exactly as informative as it was.
- Do not add anything new.
- If removing the unsupported claims leaves nothing substantive, reply with \
EXACTLY: "{refusal}" and nothing else.
- Otherwise -- if any substantive, supported statement remains -- return that \
statement ALONE. Never combine a real answer with the sentence "{refusal}"; \
that sentence means "no answer at all" and contradicts a partial answer sitting \
next to it.

Return only the corrected answer text."""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# A markdown block line: a table row, a list item, a heading, or a quote.
#
# Such a line is pruned as ONE unit. Splitting a table row on sentence
# boundaries and rejoining the survivors with spaces turns a table into a
# paragraph of stray pipe characters, and dropping half a row leaves a row with
# the wrong number of cells. Both are formatting altering the answer, which is
# the one thing Part A forbids outright -- so remediation operates on blocks
# once answers contain blocks.
_BLOCK_LINE_RE = re.compile(r"^\s*(?:\|.*\|\s*$|[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+)")

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def _split_units(text: str) -> list[list[str]]:
    """Split ``text`` into lines, each line into independently prunable units.

    One list per output line: a block line yields exactly one unit (the whole
    line), a run of prose lines yields the sentences of the joined run, and a
    blank line yields an empty list so paragraph breaks survive the round trip
    through :func:`_join_units`.

    Consecutive prose lines are joined BEFORE sentence-splitting, which keeps
    parity with the pre-Phase-4 behaviour: the old splitter ran over the whole
    answer, and its ``\\s+`` matched a newline, so a hard-wrapped sentence stayed
    one sentence. Splitting strictly per line would cut such a sentence in two
    and let remediation delete half of it.
    """
    lines: list[list[str]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            joined = " ".join(prose)
            lines.append(
                [s.strip() for s in _SENTENCE_SPLIT_RE.split(joined) if s.strip()]
            )
            prose.clear()

    for line in text.split("\n"):
        if not line.strip():
            flush_prose()
            lines.append([])
        elif _BLOCK_LINE_RE.match(line):
            flush_prose()
            lines.append([line.rstrip()])
        else:
            prose.append(line.strip())
    flush_prose()
    return lines


def _value_tokens(text: str) -> set[str]:
    """Figures and identifier-shaped tokens in ``text``, for claim matching.

    A table row states a value with almost no prose around it -- ``| Senior |
    28 days |`` -- so the "does this unit restate most of the claim?" test that
    works on a sentence never fires on a row: two content words out of a
    judge's ten-word paraphrase is 20% overlap, not 60%. Matching a row instead
    by *containment* would then delete the header too, since a header's words
    ("Band", "Annual leave") are also a subset of the claim. Requiring a shared
    figure or code separates the two: the flagged row carries the value, the
    header does not.
    """
    return {
        t.replace(",", "").rstrip(".").upper()
        for t in _NUMERIC_CLAIM_RE.findall(text) + _IDENT_CLAIM_RE.findall(text)
        if t.replace(",", "").rstrip(".")
    }


def _drop_headerless_tables(lines: list[str]) -> list[str]:
    """Drop a table left with a header row and no data rows.

    Pruning can legitimately remove every data row of a table -- each row
    carried an unsupported claim. What survives is a header promising columns
    with nothing underneath, which reads as a rendering fault rather than as a
    deliberate removal. The divider row is not counted as data: it has no words,
    so no claim can ever match it.
    """
    kept: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len([r for r in run if not _TABLE_DIVIDER_RE.match(r)]) > 1:
            kept.extend(run)
        run.clear()

    for line in lines:
        if _TABLE_ROW_RE.match(line):
            run.append(line)
            continue
        flush()
        kept.append(line)
    flush()
    return kept


def _join_units(lines: list[list[str]]) -> str:
    """Rebuild text from :func:`_split_units` output with its structure intact."""
    rendered = _drop_headerless_tables(
        [" ".join(units) if units else "" for units in lines]
    )
    # A fully-emptied block leaves a run of blank lines behind it.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(rendered)).strip()


def _regenerate_without(
    question: str,
    answer: str,
    context_text: str,
    unsupported: list[str],
    chat_model: str | None,
    temperature: float | None,
) -> str | None:
    """Rewrite ``answer`` with ``unsupported`` claims removed."""
    claims = "\n".join(f"- {c}" for c in unsupported)
    try:
        completion = _shared_openai().chat.completions.create(
            model=chat_model or config.CHAT_MODEL,
            temperature=config.CHAT_TEMPERATURE if temperature is None else temperature,
            messages=[
                {
                    "role": "system",
                    "content": _REPAIR_PROMPT.format(refusal=REFUSAL_MESSAGE),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\nQuestion: {question}\n\n"
                        f"Answer to correct:\n{answer}\n\n"
                        f"Unsupported claims to delete:\n{claims}"
                    ),
                },
            ],
        )
        return (completion.choices[0].message.content or "").strip() or None
    except openai.OpenAIError as exc:
        logger.warning("Grounding repair call failed: %s", exc)
        return None


def _prune_sentences(answer: str, unsupported: list[str]) -> tuple[str, list[str]]:
    """Excise the units carrying an unsupported claim, keeping the structure.

    The deterministic fallback when regeneration fails. Crude, but unlike another
    generation attempt it cannot introduce a NEW unsupported claim, so it is the
    safe terminal step. Matching is by content-word overlap because the judge
    paraphrases the claims it reports.

    A unit is a sentence in prose and a whole line inside a markdown block (see
    :func:`_split_units`), so removing one bad row from a table leaves a table
    with one fewer row rather than a paragraph of loose pipe characters.
    """
    lines = _split_units(answer)
    if not any(lines):
        return answer, []

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}

    claims = [(words(c), _value_tokens(c)) for c in unsupported]
    removed: list[str] = []
    kept_lines: list[list[str]] = []
    for units in lines:
        kept_units: list[str] = []
        for unit in units:
            unit_words = words(unit)
            is_block = bool(_BLOCK_LINE_RE.match(unit))
            drop = any(
                claim_words
                and (
                    # The claim is mostly restated by this unit -- the original
                    # rule, and the one that fits prose.
                    len(unit_words & claim_words) / len(claim_words) >= 0.6
                    # Or this unit is a terse block line whose words the claim
                    # covers, AND the two share the actual flagged value.
                    or (
                        is_block
                        and unit_words
                        and len(unit_words & claim_words) / len(unit_words) >= 0.6
                        and bool(_value_tokens(unit) & claim_tokens)
                    )
                )
                for claim_words, claim_tokens in claims
            )
            (removed if drop else kept_units).append(unit)
        kept_lines.append(kept_units)

    # `_join_units` can empty the result even when a unit survived -- a table
    # divider row matches no claim and is always kept, but is not content.
    pruned = _join_units(kept_lines)
    if not pruned:
        return REFUSAL_MESSAGE, removed
    return pruned, removed


def _strip_embedded_refusal(text: str) -> str:
    """Drop the fixed refusal sentence when it sits alongside real content.

    ``REFUSAL_MESSAGE`` means "no answer at all" (see ``_is_refusal``, which
    only matches when it is the WHOLE answer). A rewrite that keeps a genuine
    partial answer and also tacks the refusal sentence on as one more clause is
    self-contradictory -- observed live from the repair model, which the
    prompt tells what to do when nothing substantive survives but not what to
    do when something does. This is the code-level backstop for that prompt
    instruction, not a replacement for it.
    """
    if _is_refusal(text):
        return text
    lines = _split_units(text)
    total = sum(len(units) for units in lines)
    kept = [[u for u in units if not _is_refusal(u)] for units in lines]
    if sum(len(units) for units in kept) == total:
        return text
    # Structure-preserving rejoin, for the same reason `_prune_sentences` uses
    # it: a formatted answer must not be flattened on its way through repair.
    # An answer made up of nothing but the refusal sentence IS a refusal.
    return _join_units(kept) or REFUSAL_MESSAGE


def _repairable_claims(claims: list[str]) -> list[str]:
    """Keep only claims worth rewriting an answer over.

    The judge sometimes flags a statement about what the context does NOT contain.
    That is not a claim about the world and cannot be unsupported. Acting on such
    flags is actively harmful: remediation then deletes a legitimate hedge, and in
    one observed case regeneration dropped a correct "AES-256" along with it.
    """
    return [c for c in claims if not _signals_incomplete(c)]


def _supported_values(text: str, context_text: str) -> set[str]:
    """Identifiers and figures in ``text`` that the context DOES support."""
    values = set(_IDENT_CLAIM_RE.findall(text)) | set(_NUMERIC_CLAIM_RE.findall(text))
    stripped_context = context_text.replace(",", "")
    return {
        v.replace(",", "").rstrip(".")
        for v in values
        if v.replace(",", "").rstrip(".")
        and v.replace(",", "").rstrip(".") in stripped_context
    }


def _repair_is_safe(original: str, repaired: str, context_text: str) -> bool:
    """True if ``repaired`` kept every context-supported value from ``original``.

    A repair only improves things if it removes what the context does not
    support. If it also drops a value the context DOES support, it has traded a
    flagged answer for a less complete one -- observed live, where regenerating
    to remove a hedge also deleted the correct "AES-256" requirement.
    """
    lost = _supported_values(original, context_text) - _supported_values(
        repaired, context_text
    )
    if lost:
        logger.info("Rejecting grounding repair: it dropped supported %s", sorted(lost))
    return not lost


def _fence_unsupported(answer: str, unsupported: list[str]) -> str:
    """Append a visibly separate note naming what the answer text does NOT verify.

    The answer's own prose is left untouched -- it still reads as a normal
    answer -- but a clearly delimited block follows it, so a reader (or a UI
    that renders `answer` as-is) cannot mistake the unverified claim for a
    grounded one. This is the fallback for exactly the case where deleting the
    claim would also delete something supported: fencing keeps the supported
    majority of the answer AND is explicit about the part that is not verified,
    rather than choosing silently between the two.
    """
    claims = "; ".join(unsupported)
    return f"{answer.strip()}\n\n[Unverified -- not confirmed by the documents: {claims}]"


_COVERAGE_PROMPT = """The previous answer to this question left items out. The \
document's own index lists the items below as belonging to the set the question \
asks for.

Rewrite the answer so it covers EVERY listed item that the context supports.

Hard rules:
- Use ONLY the context passages. If the context says nothing about a listed item, \
name the item and say the documents do not cover it -- never invent a detail.
- Keep everything the previous answer got right; you are adding what it missed.
- Do not mention this instruction, the index, or that anything was missing."""


def enforce_enumeration_coverage(
    question: str,
    answer: str,
    context_text: str,
    enumeration,
    chat_model: str | None = None,
    temperature: float | None = None,
) -> tuple[str, dict | None]:
    """V3.3. Check an enumeration answer against the manifest count, and repair once.

    This is the step that eliminates **silent incompleteness**. An exhaustive answer
    returning three of seven items reads exactly like one returning all seven, and
    every other guard is looking elsewhere: grounding verification asks "is each
    claim supported?", and three correct items out of seven pass that perfectly. The
    manifest is the only component that knows there are seven.

    Returns ``(answer, report)``. The report is diagnostics only -- it never changes
    a citation, because citations come from the units that entered the prompt and
    that set is unchanged.
    """
    if enumeration is None:
        return answer, None

    missing = enumeration.missing_from(answer)
    report = {
        "term": enumeration.term,
        "expected": enumeration.expected,
        "covered": enumeration.expected - len(missing),
        "missing": missing,
        "truncated_from": enumeration.truncated_from,
        "repaired": False,
    }
    if not missing or not config.MANIFEST_COVERAGE_REPAIR:
        return answer, report

    listed = "\n".join(f"- {label}" for label in enumeration.labels)
    try:
        completion = _shared_openai().chat.completions.create(
            model=chat_model or config.CHAT_MODEL,
            temperature=temperature if temperature is not None else config.CHAT_TEMPERATURE,
            messages=[
                {"role": "system", "content": _COVERAGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nItems in this set:\n{listed}\n\n"
                        f"Previous answer:\n{answer}\n\nContext:\n{context_text}"
                    ),
                },
            ],
        )
        rewritten = (completion.choices[0].message.content or "").strip()
    except (openai.OpenAIError, IndexError) as exc:
        logger.warning("Enumeration coverage repair failed: %s", exc)
        return answer, report

    # Guarded like grounding remediation: a repair may only improve the answer.
    # A rewrite that withdrew it, or covered no more than before, is discarded --
    # trading a partly-right answer for a reworded one is not a repair.
    if not rewritten or _is_refusal(rewritten):
        return answer, report
    after = enumeration.missing_from(rewritten)
    if len(after) >= len(missing):
        return answer, report

    report.update(
        {"repaired": True, "covered": enumeration.expected - len(after), "missing": after}
    )
    return rewritten, report


def enforce_grounding(
    question: str,
    answer: str,
    context_text: str,
    grounding: Grounding,
    chat_model: str | None = None,
    temperature: float | None = None,
) -> tuple[str, Grounding]:
    """Return an answer that verification accepts, repairing it if necessary.

    Order: regenerate once with the offending claims named -> re-verify -> prune
    the offending sentences -> withdraw to the refusal if nothing substantive
    survives. ``unverified_numbers`` alone never triggers repair.
    """
    if not config.ENABLE_GROUNDING_REPAIR:
        return answer, grounding
    if grounding.faithful is not False or not grounding.unsupported_claims:
        return answer, grounding

    unsupported = _repairable_claims(grounding.unsupported_claims)
    if not unsupported:
        grounding.faithful = True
        grounding.note = (
            "Flagged claims were statements about missing coverage, not "
            "unsupported assertions; answer left unchanged."
        )
        return answer, grounding

    original_answer = answer

    for _ in range(max(1, config.MAX_GROUNDING_REPAIRS)):
        rewritten = _regenerate_without(
            question, answer, context_text, unsupported, chat_model, temperature
        )
        if not rewritten:
            break
        rewritten = _strip_embedded_refusal(rewritten)
        if not _is_refusal(rewritten) and not _repair_is_safe(
            answer, rewritten, context_text
        ):
            # The rewrite removed supported content along with the unsupported
            # claim. Fall through to sentence pruning.
            break
        if _is_refusal(rewritten):
            return REFUSAL_MESSAGE, Grounding(
                checked=True,
                faithful=True,
                note="All claims were unsupported; answer withdrawn.",
                repaired="regenerated",
                removed_claims=unsupported,
            )
        recheck = verify_grounding(rewritten, context_text)
        if recheck.faithful is not False:
            recheck.repaired = "regenerated"
            recheck.removed_claims = unsupported
            return rewritten, recheck
        next_claims = _repairable_claims(recheck.unsupported_claims)
        if not next_claims:
            recheck.faithful = True
            recheck.repaired = "regenerated"
            recheck.removed_claims = unsupported
            return rewritten, recheck
        answer, unsupported = rewritten, next_claims

    pruned, removed = _prune_sentences(original_answer, unsupported)
    if not _is_refusal(pruned) and not _repair_is_safe(
        original_answer, pruned, context_text
    ):
        # Even sentence surgery would cost supported information: the unsupported
        # claim shares a sentence with a supported fact. Historically this
        # returned the ORIGINAL answer unchanged, with the flag visible only in
        # `grounding.unsupported_claims` -- a caller that renders `answer` alone
        # (as the Streamlit/Next.js client does) shows the unverified content as
        # plain, confident prose. Feature 6 fences it in the visible text itself
        # instead of only in metadata a UI might not surface.
        grounding.note = (
            "Contains an unsupported claim that could not be removed without also "
            "losing supported detail; the flagged claim is listed."
        )
        grounding.repaired = "declined"
        if config.PARTIAL_ANSWER_FENCING_ENABLED:
            fenced = _fence_unsupported(original_answer, unsupported)
            grounding.repaired = "fenced"
            return fenced, grounding
        return original_answer, grounding

    if _is_refusal(pruned):
        return REFUSAL_MESSAGE, Grounding(
            checked=True,
            faithful=True,
            note="Unsupported claims could not be separated; answer withdrawn.",
            repaired="pruned",
            removed_claims=removed or unsupported,
        )

    final = verify_grounding(pruned, context_text)
    final.repaired = "pruned"
    final.removed_claims = removed or unsupported
    if final.faithful is False:
        final.note = "Residual unsupported content after pruning; treat with caution."
    return pruned, final


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _escalation_allowed(plan: QueryPlan) -> bool:
    """True if this plan was cheap enough that a shortfall deserves a retry."""
    return not plan.profile.decompose


def query(
    question: str,
    top_k: int | None = None,
    chat_model: str | None = None,
    temperature: float | None = None,
    collection_name: str | None = None,
    persist_directory=None,
    embed_fn=None,
    conversation_focus: str | None = None,
    conversation_context: str | None = None,
    _allow_escalation: bool = True,
    _force_type: str | None = None,
) -> RagResponse:
    """Answer ``question`` with adaptive, intent-aware retrieval-augmented generation.

    Args:
        conversation_focus: filename of the document last discussed in this
            session, used only by the optional document-routing signals/lock.
        conversation_context: a per-session running summary (see
            ``backend.memory``), inserted into the GENERATION prompt only --
            never into retrieval -- so a follow-up's pronouns resolve without
            retrieval quality depending on how well summarization went.

    Raises:
        InvalidQuestionError: blank or whitespace-only question.
        GenerationError: the answer model call failed.
        VectorStoreError: misconfiguration (e.g. embedding-model mismatch).
    """
    if not question or not question.strip():
        raise InvalidQuestionError("Question must not be empty or whitespace-only.")

    started = time.perf_counter()
    question = question.strip()

    plan = analyze(question)
    if _force_type:
        # Escalation: keep the question but adopt a wider profile and force
        # decomposition, so the retry searches differently rather than repeating
        # the attempt that just fell short.
        forced = analyze(question, use_llm=True)
        plan = QueryPlan(
            question=question,
            query_type=_force_type,
            sub_queries=forced.sub_queries or [question],
            keywords=forced.keywords or plan.keywords,
            profile=PROFILES[_force_type],
            classified_by="escalated",
        )
    if top_k is not None:
        plan.profile = replace(
            plan.profile,
            final_k=top_k,
            candidate_k=max(plan.profile.candidate_k, top_k * 4),
        )

    # ---- Addendum 2: fire the speculative SQL lookup BEFORE retrieval --------
    # Decision 1 runs here, on this thread, against the in-memory vocabulary
    # only -- no database access -- so a question that does not fire adds
    # nothing measurable. When it does fire, the work is handed to the pool
    # BEFORE the pipeline starts, which is what makes the two overlap.
    sql_probe = None
    sql_future = None
    if config.PARALLEL_SQL_LOOKUP_ENABLED:
        from backend import table_store

        sql_probe = table_store.prepare(current_user_id() or "", question)
        if sql_probe.fire:
            sql_future = _SQL_POOL.submit(table_store.execute, sql_probe)

    # Marks the start of the span the two are meant to share, so the overlap
    # below is a measurement rather than an assumption.
    parallel_started = time.perf_counter()

    retrieval: RetrievalResult = retrieve(
        plan,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embed_fn=embed_fn,
        conversation_focus=conversation_focus,
    )
    retrieved_at = time.perf_counter()

    # ---- collect the SQL side, then Decision 2 ------------------------------
    sql_result = None
    exact_facts = ""
    if sql_future is not None:
        from backend.table_store import SqlResult, render_facts

        try:
            sql_result = sql_future.result(timeout=_SQL_TIMEOUT_SECONDS)
        except Exception as exc:
            # Includes TimeoutError. The passages answer alone, which is exactly
            # the flag-off path -- a lookup that could not finish must never turn
            # an answerable question into an error.
            logger.warning("Parallel SQL lookup abandoned: %s", exc)
            sql_result = SqlResult(verdict="skipped", reason=f"abandoned: {exc}")
        exact_facts = render_facts(sql_result)

    # ---- orchestration layer (all flags default OFF) --------------------
    # Everything below consumes retrieval's output. With every flag off,
    # `units` is retrieval.units unchanged and no extra prompt text is added,
    # so the pipeline behaves exactly as it did before this layer existed.
    orchestration_intent = intent_of(plan)
    units = retrieval.units
    plan_scaffold = ""
    workflow_plan = None
    coverage_report = None

    # Feature 3: guarantee every major section of the routed document is
    # represented before generating a document-wide answer.
    if (
        config.OUTLINE_SYNTHESIS_ENABLED
        and orchestration_intent == DOCUMENT_WIDE
        and isinstance(retrieval.stages.get("documents_selected"), list)
    ):
        from backend.outline_synthesis import ensure_section_coverage

        units, coverage_report = ensure_section_coverage(
            units,
            retrieval.stages["documents_selected"],
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    context = assemble(
        units,
        max_tokens=min(plan.profile.max_context_tokens, config.MAX_CONTEXT_TOKENS),
        document_order=plan.profile.document_order,
        merge_adjacent=plan.profile.merge_adjacent,
        outline=retrieval.outline,
    )

    # Coverage must hold in what the model actually READS, not just in the
    # candidate list: the token budget runs during assembly and dropped the very
    # sections coverage had added. If any are missing, re-assemble once with a
    # budget raised just enough to seat them.
    if coverage_report is not None and coverage_report.added_sections:
        from backend.outline_synthesis import sections_missing_from_context

        dropped = sections_missing_from_context(context.units_used, units)
        if dropped:
            # Re-assembling at the SAME budget changes nothing -- the dropped
            # units are dropped again. Extend the budget by exactly what the
            # missing sections need, capped so a pathological document cannot
            # blow the context window open.
            base = min(plan.profile.max_context_tokens, config.MAX_CONTEXT_TOKENS)
            needed = sum(count_tokens(u.text) for u in dropped)
            context = assemble(
                units,
                max_tokens=min(base + needed, int(base * 1.6)),
                document_order=plan.profile.document_order,
                merge_adjacent=plan.profile.merge_adjacent,
                outline=retrieval.outline,
            )
            coverage_report.still_missing.extend(
                f"{u.source}: {u.metadata.get('section', '')}"
                for u in sections_missing_from_context(context.units_used, units)
            )

    # Feature 2: build an explicit workflow from the ranked sections, then
    # answer from that plan rather than letting the generator pick a few.
    if config.PLANNER_ENABLED and orchestration_intent == COMBINATION:
        from backend.planner import PLAN_INSTRUCTIONS, build_plan, keyed_context, render_plan

        workflow_plan = build_plan(question, context.units_used or units)
        if workflow_plan.ok:
            plan_scaffold = (
                f"{PLAN_INSTRUCTIONS}\n\nPlanned workflow:\n{render_plan(workflow_plan)}"
            )
            # Only the key map is prepended; the passages themselves, and
            # therefore the citations, are exactly what assembly produced.
            context = replace(
                context, text=keyed_context(workflow_plan, context.text)
            )

    try:
        completion = _shared_openai().chat.completions.create(
            model=chat_model or config.CHAT_MODEL,
            temperature=config.CHAT_TEMPERATURE if temperature is None else temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{_system_prompt(plan)}\n\n{plan_scaffold}"
                        if plan_scaffold
                        else _system_prompt(plan)
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(
                        question, context, conversation_context, exact_facts
                    ),
                },
            ],
        )
        answer_text = (completion.choices[0].message.content or "").strip()
    except openai.OpenAIError as exc:
        raise GenerationError(f"OpenAI chat completion failed: {exc}") from exc
    answered_at = time.perf_counter()

    diagnostics = {
        "plan": plan.to_dict(),
        "retrieval": retrieval.stages,
        "routing": retrieval.routing.to_dict() if retrieval.routing else None,
        "outline_sections": len(retrieval.outline),
        "orchestration": {
            "intent": orchestration_intent,
            "router_enabled": config.ROUTER_ENABLED,
            "planner": workflow_plan.to_dict() if workflow_plan else None,
            "outline_coverage": coverage_report.to_dict() if coverage_report else None,
        },
        "candidates_considered": retrieval.candidates_considered,
        "reranked": retrieval.reranked,
        "hybrid": retrieval.hybrid,
        "context_tokens": context.tokens,
        "context_blocks": len(context.blocks),
        "duplicates_removed": context.duplicates_removed,
        "adjacent_merges": context.merges_performed,
        "document_ordered": context.document_ordered,
        "dropped_units": context.dropped_units,
        "latency_ms": {
            "retrieval": round((retrieved_at - started) * 1000),
            "generation": round((answered_at - retrieved_at) * 1000),
        },
    }

    # Addendum 2 diagnostics. Reported even when the SQL result was DISCARDED:
    # the interesting failure is a correct lookup rejected by Decision 2, and
    # that is invisible unless the rejection and its reason are recorded.
    if sql_probe is not None:
        vector_ms = (retrieved_at - parallel_started) * 1000.0
        sql_ms = sql_result.elapsed_ms if sql_result else 0.0
        overlap_ms = 0.0
        if sql_result and sql_result.finished_at:
            overlap_ms = max(
                0.0,
                (
                    min(sql_result.finished_at, retrieved_at)
                    - max(sql_result.started_at, parallel_started)
                )
                * 1000.0,
            )
        diagnostics["sql_lookup"] = {
            "fired": bool(sql_future),
            "decision_1": sql_probe.reason,
            "entity": sql_probe.entity.resolved,
            "entity_score": round(sql_probe.entity.score, 1),
            "column": sql_probe.column.resolved,
            "column_score": round(sql_probe.column.score, 1),
            "aggregate": sql_probe.aggregate,
            "verdict": sql_result.verdict if sql_result else "not fired",
            "decision_2": sql_result.reason if sql_result else sql_probe.reason,
            "rows_returned": sql_result.rows_returned if sql_result else 0,
            "facts_used": len(sql_result.facts) if sql_result and sql_result.confident else 0,
            "timing_ms": {
                "sql": round(sql_ms, 2),
                "vector": round(vector_ms, 2),
                # The concurrency proof. Serial execution would give
                # wall == sql + vector; concurrent gives wall == max(the two),
                # and `overlap` is how much of the SQL work happened while the
                # pipeline was running.
                "wall": round((retrieved_at - parallel_started) * 1000, 2),
                "serial_would_be": round(sql_ms + vector_ms, 2),
                "overlap": round(overlap_ms, 2),
            },
        }

    # ESCALATION. A refusal from a cheap single-query plan is ambiguous: the
    # corpus may genuinely lack the answer, or the plan may have been too
    # narrow. Rather than pay for decomposition on every question, escalate
    # only after a cheap attempt refuses outright.
    #
    # `_signals_incomplete` used to trigger escalation here too, on the theory
    # that an answer admitting missing coverage deserved a wider retry. In
    # practice that regex (matches "does not provide", "not specified", ...)
    # fires on the routine hedge a well-grounded model adds to an ALREADY
    # CORRECT answer -- "The weight limit is 42 kg. The context does not
    # provide any additional information." That doubled the latency of a
    # large fraction of ordinary lookups and, worse, the forced retry runs
    # with `restrict_documents: False`, so it pulled unrelated documents into
    # context and produced confused, cross-document-flavoured answers to
    # simple single-document questions. `_signals_incomplete` is still used
    # (correctly) in `_repairable_claims` to protect a hedge sentence from
    # being misread as an unsupported factual claim -- that use is unrelated
    # and stays.
    if _is_refusal(answer_text) and _escalation_allowed(plan) and _allow_escalation:
        logger.info("Incomplete result on %s plan; escalating.", plan.query_type)
        escalated = query(
            question,
            top_k=top_k,
            chat_model=chat_model,
            temperature=temperature,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embed_fn=embed_fn,
            conversation_focus=conversation_focus,
            conversation_context=conversation_context,
            _allow_escalation=False,
            _force_type=MULTI_HOP,
        )
        escalated.diagnostics["escalated_from"] = plan.query_type
        escalated.diagnostics["latency_ms"]["total"] = round(
            (time.perf_counter() - started) * 1000
        )
        if not _is_refusal(escalated.answer):
            return escalated
        escalated.diagnostics["escalation_recovered"] = False
        if _is_refusal(answer_text):
            return escalated
        # The wider retry found nothing either, but the first attempt answered
        # part of the question -- keep that rather than downgrading to a refusal.

    if _is_refusal(answer_text):
        diagnostics["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000)
        return RagResponse(
            answer=REFUSAL_MESSAGE,
            sources=[],
            query_type=plan.query_type,
            grounding=Grounding(checked=False, note="Refusal: nothing to verify."),
            diagnostics=diagnostics,
        )

    # V3.3: completeness before faithfulness. The coverage repair can only ADD
    # items the context supports, so the grounding check that follows still runs
    # over the final text -- whereas checking coverage after grounding would leave
    # a repaired answer unverified.
    # Addendum 2: the exact fact is part of what the answer is allowed to say, so
    # it is part of what verification checks against. Without this the verifier
    # sees "78" in the answer and not in the passages, calls it an unsupported
    # number, and strips out the very value the lookup was for. The fact is not
    # generated text -- it is a cell extracted verbatim from the document, cited
    # with its own page -- so admitting it here widens the evidence, not the
    # licence to invent.
    verification_context = (
        f"{context.text}\n\n{exact_facts}" if exact_facts else context.text
    )

    answer_text, coverage = enforce_enumeration_coverage(
        question,
        answer_text,
        verification_context,
        getattr(retrieval, "enumeration", None),
        chat_model,
        temperature,
    )
    if coverage:
        diagnostics["enumeration"] = coverage

    grounding = verify_grounding(answer_text, verification_context)
    answer_text, grounding = enforce_grounding(
        question, answer_text, verification_context, grounding, chat_model, temperature
    )

    diagnostics["latency_ms"]["verification"] = round(
        (time.perf_counter() - answered_at) * 1000
    )
    diagnostics["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000)

    if _is_refusal(answer_text):
        # Remediation withdrew the answer; citations would misrepresent the
        # passages as supporting something no longer claimed.
        return RagResponse(
            answer=REFUSAL_MESSAGE,
            sources=[],
            query_type=plan.query_type,
            grounding=grounding,
            diagnostics=diagnostics,
        )

    sources = build_sources(context)
    if sql_result is not None and sql_result.confident:
        sources = add_sql_sources(sources, sql_result.facts)

    return RagResponse(
        answer=answer_text,
        sources=sources,
        query_type=plan.query_type,
        grounding=grounding,
        diagnostics=diagnostics,
    )
