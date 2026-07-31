"""Phase 3 — the core RAG query function.

This module is the heart of SmartDoc: given a plain-English question, it

1. embeds the question with the SAME OpenAI embedding model used at ingest
   time (``config.EMBED_MODEL``, via ``backend.vectorstore.openai_embed_fn``),
2. retrieves the top-k most similar chunks (with their ``{source, page,
   chunk_index}`` metadata) from the already-persisted Chroma collection
   (``backend.vectorstore.query_collection`` -- no re-embedding of the
   corpus happens here),
3. assembles a prompt containing ONLY those retrieved chunks plus a strict
   system instruction to answer solely from that context, and
4. calls ``gpt-4o-mini`` (``config.CHAT_MODEL``) at low temperature to
   produce a grounded answer.

It does not do ingestion, chunking, or embedding of documents -- those are
Phase 1/2. It does not expose an HTTP API or a UI -- those are Phase 4/5-6.

Anti-hallucination design (M6S5): the system prompt is the PRIMARY guard --
it instructs the model to answer only from the provided context and to
reply with the exact refusal string when the context doesn't contain the
answer. We deliberately do NOT use a similarity-distance threshold as a
guard: measured distances for this corpus (see the module's verification
transcript in the Phase 3 report) show in-corpus and out-of-scope queries
occupy overlapping distance ranges with only ~21 chunks across 7 short
documents, so any single cutoff would either refuse valid in-scope
questions or let out-of-scope ones through. The prompt-based guard handled
every tested case correctly without a distance floor, so no unjustified
"magic number" threshold was added.

Citations (M6S4) are built ENTIRELY from retrieval metadata and chunk text
-- never parsed out of the model's generated answer -- so they cannot be
hallucinated. See ``_build_sources``.

Decision on sources when the model declines to answer (the "I don't know"
path): ``sources`` is an EMPTY list. Rationale: sources are supposed to
mean "this is where the answer came from." If we attached the top-k
retrieved chunks anyway, a user skimming citations without reading the
refusal text could mistake them for supporting evidence for an answer that
was never actually given -- which is precisely the misleading behavior
this system exists to prevent. An empty ``sources`` list is a deliberate,
consistent signal that "no document supported this answer," which we judge
more honest than satisfying "answers include >=1 source" literally in a
case where doing so would misrepresent what was retrieved as evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import openai

from backend.config import CHAT_MODEL, TOP_K
from backend.vectorstore import openai_embed_fn, query_collection

# The exact refusal string the model must use verbatim when the retrieved
# context does not contain the answer. Comparisons against this use
# ``_is_refusal`` (normalized), not raw equality -- see that function's
# docstring for why.
REFUSAL_MESSAGE = "I don't know based on the available documents."

# Generation is kept deterministic-ish (low temperature) so the same
# question returns an equivalent answer on repeat runs (M6B1).
CHAT_TEMPERATURE = 0.0

# Length (in characters) of the excerpt/window shown for each cited source.
# Long enough to give a human a sense of the passage, short enough to stay a
# "snippet" rather than reproducing the whole chunk. Also governs the size
# of the lexically-centered window in ``_snippet``.
SNIPPET_LENGTH = 240

# Structural filter for which retrieved hits become citations (weakness-2
# fix): keep a hit only if its distance is within this margin of the best
# (lowest-distance) hit in the batch. Justified by measured distance gaps
# on this corpus's 3 known-answer questions (real numbers from a live run
# against the persisted collection):
#   "annual leave" question:   gaps from best = 0.00, 0.19, 0.69, 0.74
#   "fault code E-01" question: gaps from best = 0.00, 0.31, 0.32, 0.43
#   "password length" question: gaps from best = 0.00, 0.19, 0.21, 0.22
# In every case the genuinely-relevant hits (same document/topic as the
# best hit) sit within ~0.2 of the best distance, while irrelevant hits
# from unrelated documents/sections jump by >=0.3-0.7. A margin of 0.30
# keeps every relevant hit observed in testing while dropping the
# off-topic ones (e.g. product_manual_widgetx.pdf noise on the fault-code
# question) -- see the Phase 3 verification transcript for the before/after.
SOURCE_DISTANCE_MARGIN = 0.30

# Words to ignore when scoring which sentence of a chunk best supports the
# question, for the lexical snippet-centering heuristic in ``_snippet``.
# Small, deliberately generic English stopword list -- not corpus-tuned.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "if", "then",
    "so", "do", "does", "did", "what", "which", "who", "whom", "whose",
    "this", "that", "these", "those", "it", "its", "as", "by", "with",
    "from", "about", "into", "over", "after", "before", "under", "above",
    "below", "between", "during", "per", "how", "many", "much", "can",
    "could", "should", "would", "will", "shall", "may", "might", "must",
    "not", "no", "yes", "i", "you", "he", "she", "we", "they", "them",
    "his", "her", "their", "our", "your", "my", "me", "us", "him",
}

SYSTEM_PROMPT = f"""You are SmartDoc, a company document assistant. You answer \
employee questions using ONLY the context passages provided below, each \
labeled with its source document and page number.

Rules (follow all of them exactly):
- Use ONLY the information in the provided context. Do not use any outside \
knowledge, training data, or assumptions, even if you are confident about \
the true answer.
- If the context fully or partially answers the question, answer it as \
completely as the context allows, in clear plain English. It is fine if \
the answer draws on more than one context passage.
- If the context does not contain information that answers the question, \
you MUST reply with EXACTLY this sentence and nothing else: \
"{REFUSAL_MESSAGE}"
- Do not guess, speculate, or fill gaps with general knowledge. If you are \
not sure the context supports the answer, prefer the refusal above answering.
- Do not mention "the context" or "the documents provided" explicitly in a \
normal answer -- just answer naturally, as if you had read the source \
material. Do not fabricate citations, page numbers, or filenames in your \
answer text -- citations are handled separately by the system.
"""


class RagError(Exception):
    """Raised for retrieval or generation failures in the RAG pipeline."""


class InvalidQuestionError(RagError):
    """Raised when the input question is empty/whitespace-only."""


class GenerationError(RagError):
    """Raised when the OpenAI chat completion call fails.

    Wraps network errors, auth errors, and rate limiting from the OpenAI
    SDK so callers (Phase 4's FastAPI layer) can map this to a clean HTTP
    error response instead of the app crashing or silently returning an
    empty answer.
    """


@dataclass
class Source:
    """A single structural citation, derived entirely from retrieval metadata.

    Never constructed from the model's generated text -- see module
    docstring for why that matters for anti-hallucination guarantees.
    """

    source: str
    page: int
    snippet: str


@dataclass
class RagResponse:
    """Return type of :func:`query`. Serializes cleanly to JSON via ``asdict``."""

    answer: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain-dict/JSON-serializable representation."""
        return {
            "answer": self.answer,
            "sources": [asdict(s) for s in self.sources],
        }


_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'-]*")
# Sentence-ish spans: runs of non-terminator characters, optionally followed
# by a terminator. Deliberately simple (no NLP dependency) -- good enough
# for a lexical heuristic over short policy/manual prose.
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def _content_words(text: str) -> set[str]:
    """Lowercased, stopword-filtered word set used for lexical overlap scoring."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _snippet(question: str, text: str, length: int = SNIPPET_LENGTH) -> str:
    """Extract a snippet of ``text`` centered on its most question-relevant part.

    Weakness-1 fix: rather than always taking the head of the chunk (which
    frequently misses the actual supporting sentence -- e.g. a chunk headed
    "3. Electrical and Network Connections" whose relevant sentence about
    fault code E-01 sits several sentences later), this scores each
    sentence-ish span of ``text`` by its lexical (stopword-filtered,
    case-insensitive) word overlap with ``question``, and returns a window
    of ``length`` characters centered on the highest-scoring span. This
    stays purely structural -- derived only from the question and the
    retrieved chunk text, never from the model's generated answer -- so it
    cannot introduce hallucinated citation content.

    If no span scores above zero (no lexical overlap at all), falls back to
    the original head-of-chunk truncation.
    """
    stripped = text.strip()
    if len(stripped) <= length:
        return stripped

    question_words = _content_words(question)
    if not question_words:
        return stripped[:length].rstrip() + "..."

    spans = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(stripped) if m.group().strip()]
    best_span = None
    best_score = 0
    for start, end in spans:
        score = len(question_words & _content_words(stripped[start:end]))
        if score > best_score:
            best_score = score
            best_span = (start, end)

    if best_span is None:
        # No lexical overlap anywhere in the chunk -- fall back to the head.
        return stripped[:length].rstrip() + "..."

    best_start, best_end = best_span
    span_len = best_end - best_start
    if span_len >= length:
        window_start, window_end = best_start, best_end
    else:
        pad_total = length - span_len
        pad_before = pad_total // 2
        window_start = max(0, best_start - pad_before)
        window_end = min(len(stripped), window_start + length)
        # Re-anchor if one side hit a boundary, so the window still uses
        # its full length budget where the text allows it.
        if window_end - window_start < length:
            window_start = max(0, window_end - length)

    excerpt = stripped[window_start:window_end].strip()
    prefix = "..." if window_start > 0 else ""
    suffix = "..." if window_end < len(stripped) else ""
    return f"{prefix}{excerpt}{suffix}"


def _filter_relevant_hits(
    hits: list[dict], margin: float = SOURCE_DISTANCE_MARGIN
) -> list[dict]:
    """Keep only hits within ``margin`` distance of the best (closest) hit.

    Weakness-2 fix: citing every retrieved chunk regardless of whether it
    actually contributed dilutes citations with retrieval noise (e.g. an
    unrelated product manual pulled in only because it shares vocabulary).
    This is a purely structural filter over retrieval distances -- see
    ``SOURCE_DISTANCE_MARGIN`` for the measured justification. The best hit
    is always kept, so an answered response never ends up with zero
    sources.
    """
    if not hits:
        return []
    best_distance = min(hit["distance"] for hit in hits)
    filtered = [hit for hit in hits if hit["distance"] <= best_distance + margin]
    return filtered or hits[:1]


def _build_sources(question: str, hits: list[dict]) -> list[Source]:
    """Build structural citations from retrieval hits (metadata + chunk text).

    Args:
        question: the original question, used only to center the lexical
            snippet window (see ``_snippet``) -- never to alter the
            source/page metadata itself.
        hits: results from ``vectorstore.query_collection``, each a dict
            with ``document`` and ``metadata`` (``source``, ``page``,
            ``chunk_index``) keys.

    Returns:
        One ``Source`` per relevant hit (see ``_filter_relevant_hits``), in
        the same (relevance) order.
    """
    sources: list[Source] = []
    for hit in _filter_relevant_hits(hits):
        meta = hit["metadata"]
        sources.append(
            Source(
                source=meta["source"],
                page=meta["page"],
                snippet=_snippet(question, hit["document"]),
            )
        )
    return sources


def _is_refusal(answer_text: str) -> bool:
    """True iff the ENTIRE answer is a cosmetic variant of ``REFUSAL_MESSAGE``.

    Weakness-3 fix: raw string equality is brittle -- a wrapped-in-quotes
    reply, a trailing newline, or different capitalization would all fail
    an exact match and route a genuine refusal down the "answered" path
    with citations attached, which is exactly the misleading output this
    system exists to prevent. This normalizes whitespace, strips a
    matching pair of surrounding quote characters, and compares
    case-insensitively.

    Deliberately distinct from substring containment: a longer, genuinely
    partial answer may include the refusal sentence as one clause among
    others (e.g. "X is four weeks. I don't know based on the available
    documents." for the other half of a two-part question) -- that is a
    real, sourced answer and must keep its citations, so this function
    returns False for it. Only when the refusal sentence *is* the whole
    answer (after stripping only cosmetic wrapping) does this return True.
    """
    normalized = re.sub(r"\s+", " ", answer_text.strip()).strip()
    quote_chars = "\"'“”‘’"
    while len(normalized) >= 2 and normalized[0] in quote_chars and normalized[-1] in quote_chars:
        normalized = normalized[1:-1].strip()
    return normalized.lower() == REFUSAL_MESSAGE.lower()


def _build_user_prompt(question: str, hits: list[dict]) -> str:
    """Assemble the user-turn prompt: labeled context blocks + the question.

    Each context block is labeled with its source and page so the model can
    ground its answer, but the model is never asked to emit citations
    itself -- citations are built separately from ``hits`` metadata.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        blocks.append(
            f"[Context {i} -- source: {meta['source']}, page: {meta['page']}]\n"
            f"{hit['document']}"
        )
    context_text = "\n\n".join(blocks) if blocks else "(no relevant context retrieved)"

    return (
        f"Context passages:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context passages above."
    )


def query(
    question: str,
    top_k: int | None = None,
    chat_model: str | None = None,
    temperature: float = CHAT_TEMPERATURE,
) -> RagResponse:
    """Answer ``question`` using retrieval-augmented generation.

    Embeds ``question`` with the same embedding model used at ingest,
    retrieves the top-k most similar chunks from the persisted Chroma
    collection, and asks ``gpt-4o-mini`` to answer strictly from that
    context. See the module docstring for the anti-hallucination and
    citation design.

    Args:
        question: the natural-language question to answer. Must not be
            empty/whitespace-only.
        top_k: number of chunks to retrieve; defaults to ``config.TOP_K``.
        chat_model: override for ``config.CHAT_MODEL`` (mainly for tests).
        temperature: generation temperature; defaults to the module's low,
            near-deterministic constant.

    Returns:
        A ``RagResponse`` with ``answer`` (str) and ``sources`` (list of
        ``Source``, empty when the model declines to answer -- see module
        docstring for that decision's rationale).

    Raises:
        InvalidQuestionError: if ``question`` is empty or whitespace-only.
        GenerationError: if the OpenAI chat completion call fails (auth,
            network, rate limit, or any other API error).
    """
    if not question or not question.strip():
        raise InvalidQuestionError("Question must not be empty or whitespace-only.")

    model = chat_model or CHAT_MODEL
    hits = query_collection(
        query_text=question, top_k=top_k or TOP_K, embed_fn=openai_embed_fn()
    )

    user_prompt = _build_user_prompt(question, hits)

    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = (completion.choices[0].message.content or "").strip()
    except openai.OpenAIError as exc:
        raise GenerationError(f"OpenAI chat completion failed: {exc}") from exc

    if _is_refusal(answer_text):
        # Normalize to the canonical refusal string so callers can rely on
        # exact-match downstream, regardless of cosmetic model variation.
        return RagResponse(answer=REFUSAL_MESSAGE, sources=[])

    return RagResponse(answer=answer_text, sources=_build_sources(question, hits))
