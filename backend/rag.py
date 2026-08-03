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
"""

from __future__ import annotations

import json
import logging
import re
import time
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
from backend.vectorstore import VectorStoreError, _shared_openai  # noqa: F401

logger = logging.getLogger("smartdoc.rag")

REFUSAL_MESSAGE = "I don't know based on the available documents."

SNIPPET_LENGTH = 260

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


def _system_prompt(plan: QueryPlan) -> str:
    """Base grounding rules plus the instructions for this query type."""
    return f"{_BASE_RULES}\n\n{_TYPE_RULES.get(plan.query_type, '')}".strip()


def build_prompt(
    question: str, context: AssembledContext, conversation_context: str | None = None
) -> str:
    """Assemble the user turn: conversation memory, labelled context, question.

    ``conversation_context`` (a per-session running summary) is placed BEFORE
    the retrieved context and labelled distinctly from it, so the system
    prompt's "memory resolves references, never supplies facts" rule has a
    visibly separate block to point at rather than an ambiguous blend.
    """
    body = context.text or "(no relevant context retrieved)"
    memory_block = (
        f"Conversation memory (for resolving references only):\n{conversation_context}\n\n"
        if conversation_context
        else ""
    )
    return (
        f"{memory_block}"
        f"Context passages:\n\n{body}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
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


def _normalise_answer(text: str) -> str:
    """Normalise for refusal comparison: whitespace, quotes, case."""
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    while len(collapsed) > 1 and collapsed[0] in _QUOTES and collapsed[-1] in _QUOTES:
        collapsed = collapsed[1:-1].strip()
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
            )
        )
    return sources


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
    # buries real hallucinations in noise.
    answer = re.sub(r"(?m)^\s*\d+[.)]\s+", "", answer)

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
    """Excise sentences carrying an unsupported claim.

    The deterministic fallback when regeneration fails. Crude, but unlike another
    generation attempt it cannot introduce a NEW unsupported claim, so it is the
    safe terminal step. Matching is by content-word overlap because the judge
    paraphrases the claims it reports.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
    if not sentences:
        return answer, []

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}

    removed: list[str] = []
    kept: list[str] = []
    claim_words = [words(c) for c in unsupported]
    for sentence in sentences:
        sentence_words = words(sentence)
        drop = any(
            claim and len(sentence_words & claim) / max(1, len(claim)) >= 0.6
            for claim in claim_words
        )
        (removed if drop else kept).append(sentence.strip())

    if not kept:
        return REFUSAL_MESSAGE, removed
    return " ".join(kept).strip(), removed


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
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    kept = [s for s in sentences if not _is_refusal(s)]
    if len(kept) == len(sentences):
        return text
    return " ".join(kept).strip()


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

    retrieval: RetrievalResult = retrieve(
        plan,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embed_fn=embed_fn,
        conversation_focus=conversation_focus,
    )
    retrieved_at = time.perf_counter()

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
                    "content": build_prompt(question, context, conversation_context),
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

    grounding = verify_grounding(answer_text, context.text)
    answer_text, grounding = enforce_grounding(
        question, answer_text, context.text, grounding, chat_model, temperature
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

    return RagResponse(
        answer=answer_text,
        sources=build_sources(context),
        query_type=plan.query_type,
        grounding=grounding,
        diagnostics=diagnostics,
    )
