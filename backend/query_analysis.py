"""Query understanding: classification, planning, and retrieval profiles.

A single fixed ``top_k`` cannot serve every question, and neither can a single
retrieval STRATEGY. "How many sick days do I get?" is answered by one chunk;
"list every fault code" needs the whole table plus codes documented in prose;
"how does leave differ between Standard and Executive bands?" needs two specific
passages that one embedding search will not both surface, because the query
embedding sits between them and the nearer section wins every slot.

So each intent selects a MODE (how evidence is found) as well as a budget:

============  =============  =========  =======  ================================
intent        mode           cand. k    final k  strategy
============  =============  =========  =======  ================================
fact_lookup   focused        20         4        few chunks, gated to top docs
comparison    per_entity     40         10       guaranteed slots per entity
multi_hop     multi_hop      40         8        bridge-first, routing NOT gated
procedural    focused        30         8        neighbours, document order
synthesis     outline        60         16       subtopic plan + section breadth
exhaustive    sweep          80         24       examine EVERY routed section
cross_doc     broad          60         14       spread across documents, ungated
============  =============  =========  =======  ================================

Cost control: the classifier is skipped when a cheap lexical test already
identifies a simple lookup, which is the common case. Only questions showing
complexity signals pay for an LLM call, so the fast path keeps its latency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import openai

import backend.config as config

FACT_LOOKUP = "fact_lookup"
COMPARISON = "comparison"
MULTI_HOP = "multi_hop"
PROCEDURAL = "procedural"
SYNTHESIS = "synthesis"
EXHAUSTIVE = "exhaustive"
CROSS_DOCUMENT = "cross_document"

QUERY_TYPES = (
    FACT_LOOKUP,
    COMPARISON,
    MULTI_HOP,
    PROCEDURAL,
    SYNTHESIS,
    EXHAUSTIVE,
    CROSS_DOCUMENT,
)

# Retrieval MODES -- how evidence is found, as opposed to how much of it.
# Previously every intent ran the same fan-out-fuse-rerank-truncate path at a
# different depth, which is why document-wide synthesis and exhaustive
# extraction failed: both are coverage problems, and top-k by similarity is the
# wrong primitive for coverage.
MODE_FOCUSED = "focused"
MODE_PER_ENTITY = "per_entity"
MODE_MULTI_HOP = "multi_hop"
MODE_OUTLINE = "outline"
MODE_SWEEP = "sweep"
MODE_BROAD = "broad"


@dataclass(frozen=True)
class RetrievalProfile:
    """Retrieval budget AND strategy for one query type."""

    query_type: str
    mode: str
    candidate_k: int
    final_k: int
    max_per_source: int
    decompose: bool
    expand_neighbours: bool
    max_context_tokens: int

    # Minimum reranker score (0-3) a candidate needs to survive. Precision-first
    # types demand 2 ("clearly relevant"); completeness-first types accept 1
    # ("might contribute"), because for those a missed item is worse than an
    # extra passage.
    min_rerank_score: float = 2.0

    # --- hierarchical routing ---
    # Whether document routing RESTRICTS retrieval (as opposed to merely
    # observing it). Intents whose evidence is spread across documents by
    # definition must not be gated, or routing removes the very passage that
    # makes the answer possible.
    restrict_documents: bool = True
    max_documents: int = 2

    # Per-intent document gate. Modes that draw HEAVILY from each document they
    # keep (outline, sweep) need a stricter gate: admitting a marginal document
    # costs them a large share of the context and, for a sweep, can force the
    # candidate set to be truncated. None means config.DOC_SCORE_DROP_RATIO.
    doc_gate_ratio: float | None = None

    # --- planning ---
    # Ask the planner for the subtopics a COMPLETE answer must cover, rather
    # than paraphrases of the question.
    plan_subtopics: bool = False

    # --- assembly ---
    # Present context in document order and merge adjacent passages instead of
    # ordering by relevance. For synthesis and procedures reading order IS
    # information; for a fact lookup it is irrelevant.
    document_order: bool = False
    merge_adjacent: bool = False


PROFILES: dict[str, RetrievalProfile] = {
    FACT_LOOKUP: RetrievalProfile(
        FACT_LOOKUP, mode=MODE_FOCUSED,
        candidate_k=20, final_k=4, max_per_source=3,
        decompose=False, expand_neighbours=False, max_context_tokens=4000,
        min_rerank_score=2.0,
        restrict_documents=True, max_documents=2,
    ),
    COMPARISON: RetrievalProfile(
        COMPARISON, mode=MODE_PER_ENTITY,
        candidate_k=40, final_k=10, max_per_source=4,
        decompose=True, expand_neighbours=False, max_context_tokens=7000,
        min_rerank_score=2.0,
        # Compared entities frequently live in different documents (employee vs
        # contractor handbook), so routing keeps more documents than a lookup.
        restrict_documents=True, max_documents=4,
    ),
    MULTI_HOP: RetrievalProfile(
        MULTI_HOP, mode=MODE_MULTI_HOP,
        candidate_k=40, final_k=8, max_per_source=3,
        decompose=True, expand_neighbours=False, max_context_tokens=7000,
        # Lenient: the bridging passage often looks only loosely relevant to the
        # question as asked, and losing it breaks the whole chain.
        min_rerank_score=1.0,
        # NOT gated. The bridging fact ("Northwind is Tier 3") sits in a
        # different document from the rule it unlocks; gating to the top
        # document would delete it.
        restrict_documents=False, max_documents=4,
    ),
    PROCEDURAL: RetrievalProfile(
        PROCEDURAL, mode=MODE_FOCUSED,
        candidate_k=30, final_k=8, max_per_source=6,
        decompose=False, expand_neighbours=True, max_context_tokens=7000,
        min_rerank_score=1.0,
        restrict_documents=True, max_documents=2,
        # Ordered steps only make sense in order, and a step split across a
        # chunk boundary must be rejoined.
        document_order=True, merge_adjacent=True,
    ),
    SYNTHESIS: RetrievalProfile(
        SYNTHESIS, mode=MODE_OUTLINE,
        candidate_k=60, final_k=16, max_per_source=10,
        # Strict, unlike other completeness-first types. A document-wide summary
        # covers many sections, so "might contribute" (1) admits the boilerplate
        # that pads every section of a real policy and crowds out its substance.
        min_rerank_score=2.0,
        decompose=True, expand_neighbours=False, max_context_tokens=9000,
        restrict_documents=True, max_documents=2, doc_gate_ratio=0.6,
        plan_subtopics=True,
        document_order=True, merge_adjacent=True,
    ),
    EXHAUSTIVE: RetrievalProfile(
        EXHAUSTIVE, mode=MODE_SWEEP,
        candidate_k=80, final_k=24, max_per_source=24,
        decompose=True, expand_neighbours=True, max_context_tokens=10000,
        min_rerank_score=1.0,
        # Stricter gate: a sweep reads EVERY chunk of every document it keeps, so
        # a marginal second document both pollutes the context and pushes the
        # candidate set past the sweep cap, forcing truncation of the very
        # completeness the mode exists to guarantee.
        restrict_documents=True, max_documents=2, doc_gate_ratio=0.65,
        document_order=True, merge_adjacent=True,
    ),
    CROSS_DOCUMENT: RetrievalProfile(
        CROSS_DOCUMENT, mode=MODE_BROAD,
        candidate_k=60, final_k=14, max_per_source=4,
        decompose=True, expand_neighbours=False, max_context_tokens=9000,
        min_rerank_score=1.0,
        # Explicitly cross-document: gating would defeat the request.
        restrict_documents=False, max_documents=6,
        plan_subtopics=True,
    ),
}


@dataclass
class QueryPlan:
    """The analysed question and the plan for retrieving its answer."""

    question: str
    query_type: str
    sub_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    profile: RetrievalProfile = PROFILES[FACT_LOOKUP]
    classified_by: str = "heuristic"  # heuristic | llm | fallback | escalated

    # Subtopics a complete answer must cover (synthesis / cross-document).
    # Distinct from sub_queries: a sub-query is a rephrasing used to find one
    # fact, whereas a subtopic is a DIMENSION of the answer that has to be
    # covered whether or not the question named it.
    subtopics: list[str] = field(default_factory=list)

    # Entities to be compared, each of which gets guaranteed retrieval slots.
    entities: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.profile.mode

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "mode": self.profile.mode,
            "sub_queries": self.sub_queries,
            "subtopics": self.subtopics,
            "entities": self.entities,
            "keywords": self.keywords,
            "classified_by": self.classified_by,
            "final_k": self.profile.final_k,
            "candidate_k": self.profile.candidate_k,
            "restrict_documents": self.profile.restrict_documents,
            "max_documents": self.profile.max_documents,
        }


# ---------------------------------------------------------------------------
# Lexical signals
# ---------------------------------------------------------------------------

_EXHAUSTIVE_RE = re.compile(
    r"\b(all|every|each|list|enumerate|complete list|full list|group all|"
    r"how many (?:different|types|kinds)|what are the)\b",
    re.I,
)
# Stems, not exact words: "compared" and "differs" are the forms people actually
# type, and \bcompare\b matches neither. A comparison that slips through here is
# classified as a plain lookup and retrieves only one side -- observed live.
_COMPARISON_RE = re.compile(
    r"\b(compar\w*|differ\w*|versus|vs\.?|between .+ and |more than|less than|"
    r"higher than|lower than|instead of|as opposed to|relative to)\b",
    re.I,
)
_PROCEDURAL_RE = re.compile(
    r"\b(how do i|how does one|how to|steps|procedure|process for|walk me "
    r"through|what is the process|in order to)\b",
    re.I,
)
_SYNTHESIS_RE = re.compile(
    r"\b(summar\w+|overview|explain everything|explain the (?:entire|whole)|"
    r"design a (?:complete )?workflow|brief me|what do i need to know|"
    r"across (?:all|the) (?:docs|documents|policies))\b",
    re.I,
)
_CROSS_DOC_RE = re.compile(
    r"\b(across (?:all|our|the) (?:documents|policies|handbooks)|"
    r"conflict\w*|inconsisten\w*|reconcile|both (?:documents|policies))\b",
    re.I,
)
_MULTI_HOP_RE = re.compile(
    r"\b(therefore|because of|given that|based on (?:its|their)|which means|"
    r"as a result of)\b",
    re.I,
)

# Feature 7 (PLANNER_INTENT_EXPANSION_ENABLED) -- naming a whole-document
# artifact type is treated as asking about the WHOLE document, even without an
# explicit "summarize"/"overview" cue that _SYNTHESIS_RE requires. "What's in
# the onboarding guide?" and "walk me through the vendor SOP" both name an
# artifact rather than a fact, so a plain lookup profile under-retrieves them.
_ARTIFACT_WIDE_RE = re.compile(
    r"\b(guide|playbook|checklist|manual|sop|onboarding|policy|journey|"
    r"timeline)\b",
    re.I,
)

# Feature 8 (EXHAUSTIVE_TRIGGER_ENABLED) -- a hard override applied AFTER
# classification, unlike _EXHAUSTIVE_RE above which only gates the fast
# heuristic path (skipped whenever the LLM classifier runs, which is most
# non-trivial questions). "everything" is added as its own alternative: \bevery\b
# requires a word boundary right after "every", which "everything" does not
# have, so "everything provided" was previously invisible to this signal.
_FORCE_EXHAUSTIVE_RE = re.compile(
    r"\b(all|every|each|everything|list|enumerate|complete list|full list|"
    r"group all)\b",
    re.I,
)

# Rare, high-signal tokens that dense embeddings handle poorly: fault codes,
# standards, tiers, versions. These are where keyword search wins.
_RARE_TOKEN_RE = re.compile(
    r"\b(?:[A-Z]{1,4}-\d{1,4}|[A-Z]{2,}[-\s]?\d{2,4}|Tier\s?\d|P\d|"
    r"TLS\s?\d(?:\.\d)?|AES-?\d{3}|SOC\s?\d)\b",
    re.I,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "its", "many", "much",
    "must", "of", "on", "or", "our", "that", "the", "their", "there", "these",
    "this", "to", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}

# Verbs describing the REQUEST rather than the subject; including them in a
# subtopic query dilutes it.
_SUBTOPIC_STOP = {
    "design", "summarise", "summarize", "explain", "describe", "outline", "give",
    "provide", "complete", "entire", "whole", "overview", "workflow", "process",
    "steps", "list", "extract", "group", "all", "every",
}

# A capitalised token that is not sentence-initial and not a common word is
# usually a named entity. Questions about a named entity are the ones most
# likely to need an intermediate lookup, so they must NOT take the
# no-classifier fast path: skipping classification there produced a real
# failure, where the answer asserted a tier that was never retrieved.
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
_COMMON_CAPITALISED = {
    "I", "Acme", "HR", "IT", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November",
    "December",
}

# English function words. A question containing none of them is almost certainly
# not English, and the corpus is English -- so it must be translated before
# retrieval. Without this, cross-lingual queries get no help from the keyword
# retriever and dense retrieval alone picks the wrong document: observed live,
# where a Spanish question about Standard-band leave retrieved the CONTRACTOR
# handbook and answered about contractors.
_ENGLISH_FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did",
    "how", "what", "which", "who", "when", "where", "why", "many", "much", "of",
    "for", "to", "in", "on", "at", "and", "or", "not", "can", "must", "should",
    "would", "will", "have", "has", "get", "gets", "there", "their", "my", "our",
    "i", "we", "you", "if", "than", "then", "with", "from",
}


def extract_keywords(question: str) -> list[str]:
    """Extract high-signal lexical terms for the keyword retriever.

    Rare identifier-shaped tokens are kept verbatim (case included) because
    "E-01" and "e 01" are different terms to BM25; ordinary content words are
    lowercased and stopword-filtered.
    """
    rare = [m.group(0) for m in _RARE_TOKEN_RE.finditer(question)]
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", question)
        if w.lower() not in _STOPWORDS and len(w) > 2
    ]
    seen: set[str] = set()
    out: list[str] = []
    for term in rare + words:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def looks_english(question: str) -> bool:
    """Heuristic: does this question contain English function words?

    Deliberately crude. A false negative costs one extra classifier call; a
    false positive searches a cross-lingual query in the wrong language and
    retrieves the wrong document entirely.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", question)]
    if len(tokens) < 4:
        return True
    return any(token in _ENGLISH_FUNCTION_WORDS for token in tokens)


def mentions_entity(question: str) -> bool:
    """True if the question names a specific entity needing resolution."""
    words = question.split()
    for match in _ENTITY_RE.finditer(question):
        token = match.group(0)
        if words and token.split()[0] == words[0].strip("?,."):
            continue  # sentence-initial capital is just sentence case
        if token in _COMMON_CAPITALISED:
            continue
        return True
    return False


def _subject_terms(question: str, limit: int = 6) -> str:
    """Extract the question's subject words, to scope subtopic searches.

    A bare subtopic like "testing" matches every document that mentions testing.
    Prefixing the question's own content words keeps each subtopic search
    anchored to the subject, which is what stops document-wide synthesis from
    wandering across the corpus.
    """
    words = [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", question)
        if w.lower() not in _STOPWORDS
        and w.lower() not in _SUBTOPIC_STOP
        and len(w) > 2
    ]
    return " ".join(words[:limit])


def heuristic_type(question: str) -> str | None:
    """Classify by lexical signal, or return None if ambiguous."""
    if _CROSS_DOC_RE.search(question):
        return CROSS_DOCUMENT
    if _EXHAUSTIVE_RE.search(question):
        return EXHAUSTIVE
    if _COMPARISON_RE.search(question):
        return COMPARISON
    if _SYNTHESIS_RE.search(question):
        return SYNTHESIS
    if _PROCEDURAL_RE.search(question):
        return PROCEDURAL
    if _MULTI_HOP_RE.search(question):
        return MULTI_HOP
    return None


def _looks_simple(question: str) -> bool:
    """True if the question is short, single-clause, unmarked, and entity-free."""
    if len(question) > 140:
        return False
    if question.count("?") > 1 or " and " in question.lower():
        return False
    if mentions_entity(question):
        return False
    if not looks_english(question):
        return False  # needs translation, which only the classifier does
    return heuristic_type(question) is None


# ---------------------------------------------------------------------------
# LLM classification + planning
# ---------------------------------------------------------------------------

_CLASSIFIER_PROMPT = """You classify document-search questions and plan their \
retrieval. Reply with JSON only.

Types:
- fact_lookup: one specific fact from one place.
- comparison: contrasts two or more named things; each side must be found \
separately.
- multi_hop: the answer needs an intermediate fact first. In particular, ANY \
question that names a specific entity (a vendor, a record type, a product, a \
band) and asks about a rule that depends on that entity's category, tier, or \
classification is multi_hop -- the category must be looked up before the rule \
can be applied. Classify these as multi_hop even when they read like a single \
simple question.
- procedural: asks for ordered steps or a process.
- synthesis: asks for a DOCUMENT-WIDE treatment of one subject -- summarise the \
document, explain the entire process, design a complete workflow, give an \
overview. The answer must cover a whole document's worth of material.
- exhaustive: asks for ALL items of some kind ("extract every", "list all", \
"group all"); completeness is the point.
- cross_document: explicitly requires reasoning ACROSS several documents -- \
reconciling, aggregating, or checking consistency between them. Use only when \
more than one document must be consulted BY DESIGN.

Rules for sub_queries:
- Write 2-4 standalone search queries ONLY when the question genuinely needs \
separate lookups (comparison, multi_hop, synthesis, exhaustive, \
cross_document). For fact_lookup and procedural, return an empty list.
- Each sub-query must be self-contained: no pronouns, no "it", no "that one".
- For multi_hop, the FIRST sub-query must retrieve the bridging fact.
- ALWAYS write sub_queries and keywords in ENGLISH: the corpus is English.
- OVERRIDING RULE: if the question is NOT in English, sub_queries must contain \
its faithful English translation even for fact_lookup and procedural. This \
overrides the "empty list" rule -- the translation is the only searchable form \
of the question.

Rules for subtopics (synthesis and cross_document ONLY, else empty):
- List 4-8 subtopics a COMPLETE answer must cover. A subtopic is a DIMENSION of \
the answer, not a rephrasing of the question -- include dimensions the question \
did not name but that the subject obviously requires.
- Derive them from the subject matter itself so they work for any document \
collection. For "design a workflow for implementing a feature": planning, \
understanding existing code, implementation, refactoring, performance, testing, \
best practices, deployment.
- Keep each 1-4 words, phrased as a searchable topic, not a question.

Rules for entities (comparison ONLY, else empty):
- Name each thing being compared, as a document would refer to it.

Reply exactly:
{"type": "<one type>", "sub_queries": ["..."], "subtopics": ["..."],
 "entities": ["..."], "keywords": ["..."]}"""


def _classify_with_llm(question: str) -> dict | None:
    """Classify, decompose, and plan via one cheap LLM call.

    Returns None if the call or parse fails -- callers fall back to heuristics
    rather than erroring, because a classification failure should degrade
    retrieval quality, not break the request.
    """
    try:
        from backend.vectorstore import _shared_openai

        completion = _shared_openai().chat.completions.create(
            model=config.UTILITY_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except (openai.OpenAIError, json.JSONDecodeError, KeyError, IndexError):
        return None

    query_type = str(payload.get("type", "")).strip().lower()
    if query_type not in QUERY_TYPES:
        return None

    def _clean(key: str, limit: int) -> list[str]:
        return [
            str(item).strip() for item in (payload.get(key) or []) if str(item).strip()
        ][:limit]

    return {
        "type": query_type,
        "sub_queries": _clean("sub_queries", 4),
        "subtopics": _clean("subtopics", 8),
        "entities": _clean("entities", 6),
        "keywords": _clean("keywords", 12),
    }


def _apply_lexical_overrides(plan: QueryPlan) -> QueryPlan:
    """Apply Features 7/8's post-classification overrides, both OFF by default.

    Runs at every ``analyze()`` return point -- including the fast heuristic
    path -- rather than living inside ``heuristic_type()``, because that
    function only gates the fast path and the FALLBACK branch; the common case
    for any non-trivial question is the LLM classifier, whose result these
    flags need to override too, not just the heuristic's.

    Order: exhaustive first, since a question can contain both an artifact
    word and an exhaustiveness marker ("list every SOP requirement"), and
    completeness is the stronger of the two claims -- upgrading to synthesis
    afterward would be a downgrade.
    """
    question = plan.question
    if (
        config.EXHAUSTIVE_TRIGGER_ENABLED
        # Only upgrades FROM a plain lookup/procedure, exactly like Feature 7
        # below -- never downgrades a classification that already carries its
        # own, different retrieval requirement. Measured without this guard:
        # forcing the exhaustive profile's gated sweep (restrict_documents=True,
        # max_documents=2) onto a MEASURED, REPRODUCIBLE cross_document question
        # ("xdoc-leave-across-policies") that needs cross_document's deliberately
        # UNGATED retrieval broke it outright, and onto a synthesis question
        # ("syn-leave-overview") that needs outline mode's section breadth
        # produced a less complete answer. Comparison and multi_hop are excluded
        # for the identical reason Feature 5's lock excludes them (see
        # backend.doc_router.detect_lock): their evidence needs a strategy this
        # override does not provide.
        and plan.query_type in (FACT_LOOKUP, PROCEDURAL)
        and _FORCE_EXHAUSTIVE_RE.search(question)
    ):
        plan.query_type = EXHAUSTIVE
        plan.profile = PROFILES[EXHAUSTIVE]
        plan.classified_by += "+exhaustive_trigger"
        return plan

    if (
        config.PLANNER_INTENT_EXPANSION_ENABLED
        and plan.query_type in (FACT_LOOKUP, PROCEDURAL)
        and _ARTIFACT_WIDE_RE.search(question)
    ):
        plan.query_type = SYNTHESIS
        plan.profile = PROFILES[SYNTHESIS]
        plan.classified_by += "+artifact_expansion"

    return plan


def analyze(question: str, use_llm: bool | None = None) -> QueryPlan:
    """Analyse ``question`` and return its retrieval plan."""
    question = question.strip()
    lexical_keywords = extract_keywords(question)

    if _looks_simple(question):
        return _apply_lexical_overrides(
            QueryPlan(
                question=question,
                query_type=FACT_LOOKUP,
                sub_queries=[question],
                keywords=lexical_keywords,
                profile=PROFILES[FACT_LOOKUP],
                classified_by="heuristic",
            )
        )

    allow_llm = config.ENABLE_DECOMPOSITION if use_llm is None else use_llm
    if allow_llm:
        result = _classify_with_llm(question)
        if result:
            query_type = result["type"]
            profile = PROFILES[query_type]
            # A non-English question keeps its translated sub-queries even for
            # types that do not otherwise decompose -- the translation IS the
            # searchable form, so discarding it searches the wrong language.
            needs_translation = not looks_english(question)
            queries = (
                result["sub_queries"]
                if ((profile.decompose or needs_translation) and result["sub_queries"])
                else []
            )

            subtopics = result["subtopics"] if profile.plan_subtopics else []
            if subtopics:
                subject = _subject_terms(question)
                queries = queries + [
                    f"{subject} {topic}".strip() for topic in subtopics
                ]

            entities = result["entities"] if query_type == COMPARISON else []

            # The original question always stays in the query set: a
            # decomposition that drops nuance should not also lose the phrasing
            # the user actually used.
            if question not in queries:
                queries = [question] + queries

            merged = lexical_keywords + [
                k
                for k in result["keywords"]
                if k.lower() not in {x.lower() for x in lexical_keywords}
            ]
            return _apply_lexical_overrides(
                QueryPlan(
                    question=question,
                    query_type=query_type,
                    sub_queries=queries,
                    keywords=merged,
                    profile=profile,
                    classified_by="llm",
                    subtopics=subtopics,
                    entities=entities,
                )
            )

    fallback = heuristic_type(question) or FACT_LOOKUP
    return _apply_lexical_overrides(QueryPlan(
        question=question,
        query_type=fallback,
        sub_queries=[question],
        keywords=lexical_keywords,
        profile=PROFILES[fallback],
        classified_by="fallback",
    ))
