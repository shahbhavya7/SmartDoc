"""Addendum 2: exact table values, without ever losing context.

The problem
-----------
Dense retrieval finds a table by what it is ABOUT; it cannot find a cell by what
it CONTAINS. Ask "what is Employee 86's salary band" of a 200-row register and
the embedding of the question is nearly equidistant from every row -- the rows
differ by two tokens each, and the one that matters is a token the embedder has
no reason to weight. A row in the MIDDLE of a long table is the worst case: it is
neither near the header (which the summary chunk carries) nor near the end.

What this module does NOT do
----------------------------
It does not replace retrieval. An earlier design routed cell-shaped questions to
SQL *instead of* the pipeline, and that trades one failure for another: "tell me
about Employee 86's performance" would come back as a bare number with no
surrounding policy, and any question the router mis-classified would lose its
context entirely.

So vector retrieval ALWAYS runs. SQL runs speculatively alongside it, and its
result is used only when it is confidently correct. The two decisions are
deliberately asymmetric:

* **Decision 1 -- fire SQL?** PERMISSIVE, made before any result exists. A local
  indexed read costs about a millisecond and no API call, and it runs on another
  thread, so a wrong guess costs nothing measurable. Fire on ANY hint: a token
  that fuzzy-matches a column name, a token that fuzzy-matches a row label, or a
  numeric/comparative cue word.
* **Decision 2 -- trust the result?** STRICT, made after both results are in. A
  wrong value stated as authoritative is worse than no value: the reader has no
  way to tell it apart from a right one. Requires a strong match on BOTH terms,
  exactly one row back, and no cue that the question wanted more than one answer.

Anything short of that is discarded silently and the passages answer alone, which
is precisely the pre-Addendum behaviour -- so a discarded SQL result is a no-op,
not a degradation.

Merge, don't choose
-------------------
When SQL is confident, the model gets BOTH: the exact fact, labelled
authoritative, and the retrieved passages. That is what lets one answer be both
exact and explained -- "Bhavya scored 78 in English, above the 40-mark passing
threshold" needs the cell for the 78 and the passage for the threshold.

Provenance
----------
Every cell carries the filename, page, and table title it came from, so a
SQL-path answer is cited exactly like a passage-path one. The value itself is the
extracted cell verbatim -- the same PyMuPDF extraction the chunks come from, not
anything generated -- which is what keeps the "answers come from real document
text" rule intact.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

import backend.config as config
from backend import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation and tokenising
# ---------------------------------------------------------------------------

# Tokens keep internal hyphens, dots, slashes and apostrophes, because table
# labels are full of them -- "E-04", "R-2.1", "Q1/Q2", "O'Brien". Splitting those
# apart would make the label unmatchable by the very n-gram that names it.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._'\-/%]*")

# Trailing punctuation a header picks up from the PDF ("Salary Band:", "Score*").
_TRIM_RE = re.compile(r"^[\W_]+|[\W_]+$")

_POSSESSIVE_RE = re.compile(r"'s$|'$")

# Digit runs. See ``_same_digits``: these are the part of a label that fuzzy
# matching must NOT be allowed to smooth over.
_DIGIT_RUN_RE = re.compile(r"\d+")

MAX_NGRAM = 4


def normalise(text: str) -> str:
    """Casefold, drop brackets, collapse whitespace, strip edge punctuation.

    Brackets become spaces rather than being kept: a header printed as
    "Unit cost (USD)" is asked about as "unit cost usd" or just "unit cost", and
    a retained "(" would sit between the two words in the stored key and cost
    every such question a few points of fuzzy score for nothing.
    """
    plain = re.sub(r"[()\[\]{}]", " ", (text or "").replace("\n", " "))
    collapsed = re.sub(r"\s+", " ", plain).strip()
    return _TRIM_RE.sub("", collapsed).casefold()


@dataclass(frozen=True)
class Span:
    """One candidate phrase from the question, with its token positions.

    Positions matter: the entity and the column must be resolved from DIFFERENT
    parts of the question. Without that, "what is the salary band" happily
    resolves "salary band" as both the row label and the column name and then
    looks up a cell that cannot exist.
    """

    text: str
    start: int
    end: int  # exclusive

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


def spans(question: str, max_n: int = MAX_NGRAM) -> list[Span]:
    """Every 1..max_n word window of the question, longest first.

    Longest first so a tie between "salary" and "salary band" resolves to the
    more specific one -- the shorter window is a prefix of the longer and scores
    nearly as well against it, so ordering is what decides.
    """
    # The possessive is stripped, not kept: a question names a row as
    # "Employee 45's grade", and "45's" would never equal the stored "45".
    tokens = [
        _POSSESSIVE_RE.sub("", t) for t in _TOKEN_RE.findall((question or "").casefold())
    ]
    tokens = [t for t in tokens if t]
    out: list[Span] = []
    for size in range(min(max_n, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - size + 1):
            out.append(Span(" ".join(tokens[start : start + size]), start, start + size))
    return out


# A deliberately small synonym map. It exists for the cases fuzzy matching cannot
# reach -- "marks" and "score" share no characters, so no edit-distance scorer
# will ever connect them -- and NOT as a general thesaurus. Prefix and typo cases
# ("eng" -> "English", "departmnt") are already handled by the fuzzy scorer, so
# listing them here would only add ways to match the wrong thing.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "marks": ("score", "grade", "result"),
    "mark": ("score", "grade"),
    "score": ("marks", "grade", "result"),
    "scores": ("marks", "grade"),
    "grade": ("score", "marks"),
    "result": ("score", "marks"),
    "pay": ("salary", "compensation"),
    "salary": ("pay", "compensation"),
    "dept": ("department",),
    "deadline": ("due date", "due"),
    "due": ("deadline",),
    "meaning": ("description", "definition"),
    "action": ("required action", "remedy", "fix"),
    "cost": ("price", "amount", "rate"),
    "price": ("cost", "amount", "rate"),
    "limit": ("threshold", "maximum"),
}


def _variants(text: str) -> tuple[str, ...]:
    """The phrase plus its synonym forms, deduplicated."""
    out = [text]
    for word, alternatives in SYNONYMS.items():
        if word == text:
            out.extend(alternatives)
        elif f" {word} " in f" {text} ":
            out.extend(text.replace(word, alt) for alt in alternatives)
    return tuple(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Extraction: LogicalTable -> cells
# ---------------------------------------------------------------------------


_ID_CODE_RE = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{2,10}$")


def _looks_like_id_values(values: list[str]) -> bool:
    """Is this column almost entirely short alnum codes ("EMP0034", "E-01")?

    Distinguishes an identifier column from a free-text one (a name, a
    description) using shape alone -- no header-name guessing, since headers
    vary too much across documents to rely on.
    """
    sample = [v.strip() for v in values if v and v.strip()][:50]
    if len(sample) < 3:
        return False
    hits = sum(1 for v in sample if _ID_CODE_RE.match(v))
    return hits / len(sample) >= 0.9


def cells_from_tables(tables: list, max_cells: int | None = None) -> list[dict]:
    """Flatten stitched tables into (entity, column, value) rows.

    The FIRST column is taken as the row label. That is not a guess about
    semantics -- it is how tables are written: the leftmost column names the
    thing each row is about, and every other column is an attribute of it. A
    table whose first column is itself an attribute (a bare index number) simply
    yields entities nobody asks for, and Decision 2's match floor discards them.

    A row shorter than the header is zipped to the shorter of the two rather than
    padded: a missing trailing cell is missing, and storing "" for it would let a
    lookup return an empty string as a confident exact value.

    A second row label, aliased: a table whose first column is an ID code
    ("Employee ID") and whose second is free text ("Name") names each row two
    ways, and a question can use either -- "MP0075's department" or "Employee
    75's department" must resolve to the same row. When column 0 looks like an
    ID column and column 1 does not, column 1's value is stored as a second,
    equally real row label for the same row, with every other column
    (including column 0 itself) as its attribute -- not a fallback path, a
    second entry alongside the first.
    """
    limit = max_cells if max_cells is not None else config.PARALLEL_SQL_MAX_CELLS_PER_DOC
    out: list[dict] = []

    for table in tables:
        headers = [str(h or "").strip() for h in getattr(table, "headers", []) or []]
        if len(headers) < 2:
            # One column is a list, not a table: there is no attribute to look up.
            continue
        entity_column = headers[0]
        title = getattr(table, "section", "") or entity_column
        rows = getattr(table, "rows", []) or []
        row_pages = getattr(table, "row_pages", []) or []

        alias_index = None
        if len(headers) > 2:
            col0 = [str(r[0]) if r else "" for r in rows]
            col1 = [str(r[1]) if len(r) > 1 else "" for r in rows]
            if _looks_like_id_values(col0) and not _looks_like_id_values(col1):
                alias_index = 1

        for index, row in enumerate(rows):
            values = [str(c or "").replace("\n", " ").strip() for c in row]
            if not values:
                continue
            page = row_pages[index] if index < len(row_pages) else getattr(
                table, "page_start", 0
            )

            def emit(entity: str, entity_norm: str, skip_index: int | None) -> bool:
                """Returns False once ``limit`` is hit, so the caller can stop."""
                for col_idx, (column, value) in enumerate(zip(headers, values)):
                    if col_idx == 0 and skip_index is None:
                        continue  # primary pass: column 0 IS the entity, not an attribute
                    if col_idx == skip_index:
                        continue  # alias pass: the alias column IS the entity here
                    column_norm = normalise(column)
                    if not column_norm or not value.strip():
                        continue
                    out.append(
                        {
                            "table_id": getattr(table, "table_id", ""),
                            "source": getattr(table, "source", ""),
                            "table_title": title,
                            "page": int(page or 0),
                            "row_index": index,
                            "row_entity": entity,
                            "row_entity_norm": entity_norm,
                            "column_name": column,
                            "column_norm": column_norm,
                            "value": value,
                        }
                    )
                    if len(out) >= limit:
                        logger.warning(
                            "table cell cap %d reached; later cells not stored", limit
                        )
                        return False
                return True

            entity = values[0]
            entity_norm = normalise(entity)
            if entity_norm and not emit(entity, entity_norm, skip_index=None):
                return out

            if alias_index is not None and alias_index < len(values):
                alias_entity = values[alias_index]
                alias_norm = normalise(alias_entity)
                if alias_norm and alias_norm != entity_norm:
                    if not emit(alias_entity, alias_norm, skip_index=alias_index):
                        return out
    return out


def store_document_tables(
    user_id: str, document_id: str, tables: list, doc_title: str = ""
) -> int:
    """Persist one document's cells and invalidate the owner's cached vocabulary.

    No-op without an owner row: cells are per-user by construction, and an
    unscoped ingest (the evaluation harness) has no account to attach them to.
    """
    if not config.PARALLEL_SQL_LOOKUP_ENABLED or not user_id or not document_id:
        return 0
    cells = cells_from_tables(tables)
    # Called even when `cells` is empty: a document that LOST its table on
    # re-ingest must lose its stored cells too, and only the delete inside
    # replace_table_cells does that.
    written = db.replace_table_cells(user_id, document_id, cells)
    invalidate(user_id)
    return written


# ---------------------------------------------------------------------------
# The in-memory vocabulary cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableInfo:
    table_id: str
    source: str
    title: str
    page: int
    columns: tuple[str, ...]
    entities: tuple[str, ...]


@dataclass(frozen=True)
class Vocabulary:
    """One user's column names, row labels, and table map, held in memory.

    Decision 1 fuzzy-matches against this on EVERY query, so it must not require
    a database round trip -- the whole justification for firing SQL permissively
    is that the decision itself is free.
    """

    user_id: str
    columns: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    column_display: dict[str, str] = field(default_factory=dict)
    entity_display: dict[str, str] = field(default_factory=dict)
    column_tables: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tables: dict[str, TableInfo] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.tables


_CACHE: dict[str, Vocabulary] = {}
_CACHE_LOCK = threading.Lock()


def invalidate(user_id: str | None = None) -> None:
    """Drop one user's cached vocabulary, or everyone's."""
    with _CACHE_LOCK:
        if user_id is None:
            _CACHE.clear()
        else:
            _CACHE.pop(user_id, None)


def _build_vocabulary(user_id: str) -> Vocabulary:
    rows = db.table_vocabulary(user_id)
    tables: dict[str, dict] = {}
    column_display: dict[str, str] = {}
    entity_display: dict[str, str] = {}
    column_tables: dict[str, set[str]] = {}

    for row in rows:
        table_id = row["table_id"]
        entry = tables.setdefault(
            table_id,
            {
                "source": row["source"],
                "title": row["table_title"],
                "page": int(row["page"] or 0),
                "columns": {},
                "entities": {},
            },
        )
        entry["columns"][row["column_norm"]] = row["column_name"]
        entry["entities"][row["row_entity_norm"]] = row["row_entity"]
        column_display.setdefault(row["column_norm"], row["column_name"])
        entity_display.setdefault(row["row_entity_norm"], row["row_entity"])
        column_tables.setdefault(row["column_norm"], set()).add(table_id)

    return Vocabulary(
        user_id=user_id,
        columns=tuple(sorted(column_display)),
        entities=tuple(sorted(entity_display)),
        column_display=column_display,
        entity_display=entity_display,
        column_tables={k: tuple(sorted(v)) for k, v in column_tables.items()},
        tables={
            table_id: TableInfo(
                table_id=table_id,
                source=entry["source"],
                title=entry["title"],
                page=entry["page"],
                columns=tuple(entry["columns"]),
                entities=tuple(entry["entities"]),
            )
            for table_id, entry in tables.items()
        },
    )


def vocabulary(user_id: str) -> Vocabulary:
    """This user's cached vocabulary, built on first use.

    Lazy as well as warmed at startup: ingestion, the CLI, and the tests all
    reach this without ever running the FastAPI startup hook, and a cache that
    only ever populated at boot would leave a just-uploaded document invisible to
    SQL until the next restart.
    """
    if not user_id:
        return Vocabulary(user_id="")
    with _CACHE_LOCK:
        cached = _CACHE.get(user_id)
    if cached is not None:
        return cached
    built = _build_vocabulary(user_id)
    with _CACHE_LOCK:
        _CACHE[user_id] = built
    return built


def warm_all() -> int:
    """Preload every owner who has stored cells. Called at startup and after ingest."""
    if not config.PARALLEL_SQL_LOOKUP_ENABLED:
        return 0
    users = db.users_with_tables()
    for user_id in users:
        invalidate(user_id)
        vocabulary(user_id)
    return len(users)


# ---------------------------------------------------------------------------
# Decision 1 -- fire SQL? (PERMISSIVE, before results)
# ---------------------------------------------------------------------------

# The brief's list, plus the plural/inflected forms of the same words. Nothing
# semantically new: "rates" is "rate", and leaving it out would mean the cue
# fired on one spelling of the same question and not the other.
NUMERIC_CUES = (
    "score", "scores", "scored", "marks", "mark", "grade", "grades",
    "how much", "how many", "amount", "rate", "rates", "days",
    "highest", "lowest", "total", "count", "maximum", "minimum",
    "max", "min", "top", "above", "below", "over", "under",
)

# Decision 2's veto list. A question wanting several answers must not be answered
# by one authoritative cell, however well that cell matched.
MULTI_ANSWER_CUES = (
    "all", "every", "each", "list", "lists", "compare", "comparison",
    "why", "explain", "summarize", "summarise", "summary", "breakdown",
    "overview", "tell me about", "describe",
)


def _has_cue(question: str, cues: tuple[str, ...]) -> str:
    """The first cue present as a whole word/phrase, or "".

    Whole-word, not substring: "overall" contains "all" and "counted" contains
    "count", and either would fire a rule the question never triggered.
    """
    text = f" {re.sub(r'[^a-z0-9 ]+', ' ', (question or '').casefold())} "
    text = re.sub(r"\s+", " ", text)
    for cue in cues:
        if f" {cue} " in text:
            return cue
    return ""


@dataclass
class TermMatch:
    """One resolved term: what the query said, what it maps to, how sure."""

    query_text: str = ""
    resolved: str = ""      # the normalised stored key
    display: str = ""       # the stored value as the document wrote it
    score: float = 0.0
    span: Span | None = None


# Words too common to identify anything on their own, when deciding whether a
# digit-bearing span has any REAL identifying content beyond its number.
_FILLER_WORDS = {
    "in", "on", "at", "of", "the", "a", "an", "for", "to", "is", "are", "and",
}


def _digit_match_is_meaningful(variant: str, candidate: str) -> bool:
    """Beyond sharing a number, does the query span say anything about WHICH one?

    ``_same_digits`` guarantees the two contain the same digit run, which is
    necessary but not sufficient: "70 in" and "employee 70" share the digit run
    ``["70"]`` and nothing else, and ``WRatio`` scores that pair 85.5 -- over the
    trust floor -- purely because "70" is a substantial fraction of a short
    query span. The span is a THRESHOLD ("scored above 70"), not an attempt to
    name row 70, and the giveaway is that stripping the digits leaves it with no
    real word in common with the label at all.

    Only load-bearing for candidates whose match depends on the shared digit --
    genuine word matches ("eng" -> "english") never touch this function because
    neither side has a digit to trigger the shortlist that calls it.
    """
    strip = lambda s: re.sub(r"\d+", " ", s)
    query_words = {w for w in strip(variant).split() if len(w) >= 3 and w not in _FILLER_WORDS}
    if not query_words:
        # Nothing left but digits and filler -- the number is doing all the
        # work, and a shared number alone does not name a row.
        return False
    candidate_words = strip(candidate).split()
    return any(
        w in candidate_words or any(fuzz.ratio(w, c) >= 60 for c in candidate_words)
        for w in query_words
    )


def _same_digits(left: str, right: str) -> bool:
    """Do two labels contain the same digit runs, in the same order?

    This is the guard that makes fuzzy matching safe on identifiers, and it is
    the single most important rule in this module.

    Edit distance is built for spelling variation, and a digit is not a spelling
    variation -- "Employee 45" and "Employee 1" are two different rows, not two
    renderings of one. Measured: ``WRatio("employee 45", "employee 1")`` is about
    95, comfortably over the 85 trust floor, so without this guard a question
    about row 45 of a 90-row register confidently returns row 1's salary. Every
    identifier-shaped label in this corpus has the same shape -- E-04 vs E-01,
    WGX-0042 vs WGX-0002 -- so the failure is systematic, not incidental.

    The rule is symmetric: a query with no digits does not match a label with
    digits either. The cost is a class of legitimate miss ("quarterly revenue"
    will not reach a column named "Q1 revenue"), and a miss is only a lost
    opportunity -- the passages still answer -- whereas the wrong row is a
    confidently wrong number.
    """
    return _DIGIT_RUN_RE.findall(left) == _DIGIT_RUN_RE.findall(right)


def _resolve(candidates: list[Span], choices: tuple[str, ...], display: dict) -> TermMatch:
    """Resolve a question span to a vocabulary entry: exact first, then fuzzy.

    Exact before fuzzy, because a fuzzy scorer has no notion of "this one is
    literally right" -- ``extractOne`` returns the first entry at the maximum
    score, and with 90 near-identical row labels the maximum is a tie the exact
    match does not necessarily win.

    Fuzzy uses ``WRatio``: it combines exact, partial, and token-sorted
    comparison with a length penalty, which handles both "eng" -> "english"
    (partial) and "band salary" -> "salary band" (token-sorted) without either
    needing its own rule. It runs over a digit-filtered shortlist rather than the
    whole vocabulary -- see ``_same_digits``.
    """
    best = TermMatch()
    if not choices:
        return best
    available = set(choices)

    # Pass 1: exact. ``candidates`` is longest-first, so the most specific span
    # that matches something wins -- "salary band" over "salary".
    for span in candidates:
        for variant in _variants(span.text):
            if variant in available:
                return TermMatch(
                    query_text=span.text,
                    resolved=variant,
                    display=display.get(variant, variant),
                    score=100.0,
                    span=span,
                )

    # Pass 2: fuzzy, over digit-compatible choices only.
    for span in candidates:
        for variant in _variants(span.text):
            if variant.strip().isdigit():
                # A BARE number ("70" alone, from "scored above 70") is not an
                # attempt to name a row -- it is a threshold. The digit guard
                # above checks digit runs MATCH, not that the query said
                # anything else, so "70" alone is "digit-compatible" with every
                # label containing "70" ("Employee 70") and would otherwise
                # fuzzy-resolve to one by coincidence of a shared number that
                # the question never meant as an identifier.
                continue
            shortlist = [
                c
                for c in choices
                if _same_digits(variant, c) and _digit_match_is_meaningful(variant, c)
            ]
            if not shortlist:
                continue
            hit = process.extractOne(variant, shortlist, scorer=fuzz.WRatio)
            if not hit:
                continue
            score = float(hit[1])
            # A SHORT candidate ("Bo", "E-1") scores deceptively high against an
            # unrelated longer span purely because WRatio's partial-ratio
            # component rewards the short string appearing as a substring --
            # "Bo" inside "above" scores 90, though the two share only two
            # letters in sequence. A plain edit-distance ratio does not have
            # that failure mode, so for short candidates the two are combined
            # (the lower wins): a genuine short match ("bo" query -> "Bo" row)
            # scores high on both, and a substring coincidence scores high on
            # only one.
            if len(hit[0]) <= 6:
                score = min(score, float(fuzz.ratio(variant, hit[0])))
            if score > best.score:
                best = TermMatch(
                    query_text=span.text,
                    resolved=hit[0],
                    display=display.get(hit[0], hit[0]),
                    score=score,
                    span=span,
                )
    return best


@dataclass
class Probe:
    """Decision 1's output: everything the SQL thread needs, resolved in memory.

    Built on the request thread from the cached vocabulary alone -- no database
    access -- so the fire/skip decision adds nothing measurable to a query that
    does not fire, and the worker thread starts with its terms already resolved.
    """

    user_id: str
    question: str
    fire: bool = False
    reason: str = ""
    entity: TermMatch = field(default_factory=TermMatch)
    column: TermMatch = field(default_factory=TermMatch)
    multi_answer_cue: str = ""
    numeric_cue: str = ""
    aggregate: str = ""          # "", "max", "min", "count", "filter"
    threshold: float | None = None
    comparator: str = ""         # ">" or "<"
    vocabulary_size: int = 0


# "best"/"worst" are deliberately absent: they are subjective, not numeric, and
# "who is the best engineer" is not a MAX over a column.
_SUPERLATIVE_MAX = ("highest", "greatest", "largest", "most", "top", "maximum", "max")
_SUPERLATIVE_MIN = ("lowest", "smallest", "least", "minimum", "min")
_ABOVE = ("above", "over", "more than", "greater than", "at least", "exceeds")
_BELOW = ("below", "under", "less than", "fewer than", "at most")

# Words that mark a span as belonging to a THRESHOLD or superlative phrase,
# never to an entity name. Withheld from entity resolution in ``prepare`` --
# see the comment there for the false positive this prevents.
_RESERVED_SPAN_WORDS = _SUPERLATIVE_MAX + _SUPERLATIVE_MIN + _ABOVE + _BELOW


def _detect_aggregate(
    question: str, column: TermMatch, entity: TermMatch
) -> tuple[str, str, float | None]:
    """Classify an aggregate intent: (kind, comparator, threshold).

    Two suppressions, both found by a column named ``Minimum order``:

    1. **A question that names a specific ROW is a lookup, not an aggregate.**
       "What is WGX-0045's minimum order?" contains "minimum", and without this
       it would return the numerically smallest minimum-order across all ninety
       rows -- a confident, cited answer to a question nobody asked. An aggregate
       names a SET; if the entity resolved at the trust floor, a row was named.
    2. **A superlative that is part of the column's own name is not a
       superlative.** "What is the minimum order?" has "minimum" inside the
       resolved column ``minimum order``, so the word is the column's name and
       not an instruction to rank by it.
    """
    if not config.PARALLEL_SQL_AGGREGATES_ENABLED:
        return "", "", None
    if entity.score >= config.PARALLEL_SQL_TRUST_THRESHOLD:
        return "", "", None

    # The text the column match already accounts for; a cue inside it is a name.
    named = f"{column.query_text} {column.resolved}"

    for cues, kind in ((_SUPERLATIVE_MAX, "max"), (_SUPERLATIVE_MIN, "min")):
        cue = _has_cue(question, cues)
        if cue and not _has_cue(named, (cue,)):
            return kind, "", None

    for cues, comparator in ((_ABOVE, ">"), (_BELOW, "<")):
        cue = _has_cue(question, cues)
        if not cue:
            continue
        # The number must FOLLOW the comparator phrase: "above 80" is a
        # threshold, "80 above sea level" is not.
        tail = re.split(re.escape(cue), question.casefold(), maxsplit=1)
        if len(tail) < 2:
            continue
        number = re.search(r"-?\d[\d,]*(?:\.\d+)?", tail[1])
        if number:
            kind = "count" if _has_cue(question, ("how many", "count")) else "filter"
            return kind, comparator, float(number.group(0).replace(",", ""))
    return "", "", None


def prepare(user_id: str, question: str) -> Probe:
    """Decision 1. Permissive by design: fire on any hint, discard later.

    Returns a Probe with ``fire=False`` and a reason whenever SQL is pointless,
    so the diagnostics can say *why* nothing ran rather than only that nothing did.
    """
    probe = Probe(user_id=user_id, question=question)
    if not config.PARALLEL_SQL_LOOKUP_ENABLED:
        probe.reason = "flag off"
        return probe

    vocab = vocabulary(user_id)
    probe.vocabulary_size = len(vocab.tables)
    if vocab.empty:
        # "Skip SQL entirely if the user has no tables" -- and this is also what
        # guarantees an account with no tables sees no latency change at all.
        probe.reason = "no stored tables"
        return probe

    candidates = spans(question)
    probe.column = _resolve(candidates, vocab.columns, vocab.column_display)
    # The entity must come from a DIFFERENT part of the question than the column,
    # so the column's own span is withheld from the entity search. Spans built
    # around a comparator or superlative word ("above 70", "highest") are also
    # withheld: a bare number belongs to a THRESHOLD, and matching it against a
    # numbered label ("Employee 70") by coincidence of shared digits is not the
    # question naming a row. Digit-only spans are filtered inside ``_resolve``
    # itself; this catches the multi-word spans that wrap one.
    entity_candidates = [
        s
        for s in candidates
        if not (probe.column.span and s.overlaps(probe.column.span))
        and not _has_cue(s.text, _RESERVED_SPAN_WORDS)
    ]
    probe.entity = _resolve(entity_candidates, vocab.entities, vocab.entity_display)

    probe.multi_answer_cue = _has_cue(question, MULTI_ANSWER_CUES)
    probe.numeric_cue = _has_cue(question, NUMERIC_CUES)
    # After resolution, not before: both aggregate suppressions need to know what
    # the entity and the column actually resolved to.
    probe.aggregate, probe.comparator, probe.threshold = _detect_aggregate(
        question, probe.column, probe.entity
    )

    floor = config.PARALLEL_SQL_FIRE_THRESHOLD
    reasons = []
    if probe.column.score >= floor:
        reasons.append(f"column~{probe.column.resolved}({probe.column.score:.0f})")
    if probe.entity.score >= floor:
        reasons.append(f"entity~{probe.entity.resolved}({probe.entity.score:.0f})")
    if probe.numeric_cue:
        reasons.append(f"cue:{probe.numeric_cue}")

    probe.fire = bool(reasons)
    probe.reason = ", ".join(reasons) if reasons else "no column, entity, or numeric cue"
    return probe


# ---------------------------------------------------------------------------
# Execution and Decision 2
# ---------------------------------------------------------------------------


@dataclass
class ExactFact:
    """One authoritative cell (or aggregate result), fully cited."""

    entity: str
    column: str
    value: str
    source: str
    page: int
    table_title: str
    table_id: str
    kind: str = "cell"  # "cell", "max", "min", "count", "filter"

    def line(self) -> str:
        return f"{self.entity} - {self.column}: {self.value}"

    def citation(self) -> str:
        where = f"{self.source}, page {self.page}" if self.page else self.source
        return f"Source: {where}" + (f", Table: {self.table_title}" if self.table_title else "")


@dataclass
class SqlResult:
    """What the SQL thread produced, and whether it may be shown.

    ``verdict`` is one of "confident", "ambiguous", "empty", "skipped". Only
    "confident" reaches the prompt; the rest are diagnostics, because knowing
    that SQL fired and was rejected is how a wrong rejection gets found.
    """

    verdict: str = "skipped"
    facts: list[ExactFact] = field(default_factory=list)
    reason: str = ""
    rows_returned: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def confident(self) -> bool:
        return self.verdict == "confident" and bool(self.facts)

    @property
    def elapsed_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000.0


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _as_number(value: str) -> float | None:
    """Leading number in a cell, or None. "78%" is 78; "N/A" is None.

    Parsed rather than cast in SQL because SQLite's ``CAST('N/A' AS REAL)`` is
    0.0, which would silently crown a non-numeric cell the winner of a MIN.
    """
    match = _NUMBER_RE.search(value or "")
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _fact(row: dict, kind: str = "cell") -> ExactFact:
    return ExactFact(
        entity=row["row_entity"],
        column=row["column_name"],
        value=row["value"],
        source=row["source"],
        page=int(row["page"] or 0),
        table_title=row["table_title"],
        table_id=row["table_id"],
        kind=kind,
    )


def _single_cell(probe: Probe) -> SqlResult:
    """The lookup path: one resolved entity, one resolved column, one value."""
    result = SqlResult()
    trust = config.PARALLEL_SQL_TRUST_THRESHOLD

    if not probe.entity.resolved or not probe.column.resolved:
        result.verdict = "empty"
        result.reason = "no entity/column pair resolved"
        return result

    rows = db.lookup_cells(probe.user_id, probe.entity.resolved, probe.column.resolved)
    result.rows_returned = len(rows)
    if not rows:
        result.verdict = "empty"
        result.reason = "no matching cell"
        return result

    # Decision 2, all four conditions. Evaluated after the query rather than
    # before, because "exactly one row" is only knowable once the rows are in.
    if probe.entity.score < trust:
        result.verdict = "ambiguous"
        result.reason = f"entity match {probe.entity.score:.0f} < {trust}"
    elif probe.column.score < trust:
        result.verdict = "ambiguous"
        result.reason = f"column match {probe.column.score:.0f} < {trust}"
    elif len(rows) > 1:
        result.verdict = "ambiguous"
        result.reason = f"{len(rows)} rows matched, not one"
    elif probe.multi_answer_cue:
        result.verdict = "ambiguous"
        result.reason = f"multi-answer cue '{probe.multi_answer_cue}'"
    else:
        result.verdict = "confident"
        result.facts = [_fact(rows[0])]
    return result


def _aggregate(probe: Probe) -> SqlResult:
    """MAX / MIN / COUNT / threshold filter over one resolved column.

    Vector search is genuinely poor at these -- "who scored highest in Math"
    has no passage that states the answer, because the answer is a comparison
    across rows that the document never performs.

    The extra confidence condition here, beyond Decision 2's, is that the column
    must live in exactly ONE table. "Highest score" across two unrelated score
    tables is not a fact, it is a category error.
    """
    result = SqlResult()
    trust = config.PARALLEL_SQL_TRUST_THRESHOLD

    if probe.column.score < trust:
        result.verdict = "ambiguous"
        result.reason = f"column match {probe.column.score:.0f} < {trust}"
        return result

    vocab = vocabulary(probe.user_id)
    owners = vocab.column_tables.get(probe.column.resolved, ())
    if len(owners) != 1:
        result.verdict = "ambiguous"
        result.reason = f"column '{probe.column.resolved}' spans {len(owners)} tables"
        return result

    rows = db.column_cells(probe.user_id, probe.column.resolved, table_id=owners[0])
    numeric = [(r, _as_number(r["value"])) for r in rows]
    numeric = [(r, n) for r, n in numeric if n is not None]
    result.rows_returned = len(numeric)
    if not numeric:
        result.verdict = "empty"
        result.reason = "no numeric values in column"
        return result

    if probe.aggregate in ("max", "min"):
        best = (max if probe.aggregate == "max" else min)(n for _, n in numeric)
        winners = [r for r, n in numeric if n == best]
        if len(winners) != 1:
            # A tie has no single winner, and picking one would be a fabrication.
            result.verdict = "ambiguous"
            result.reason = f"{len(winners)}-way tie for {probe.aggregate}"
            return result
        result.verdict = "confident"
        result.facts = [_fact(winners[0], kind=probe.aggregate)]
        return result

    if probe.threshold is None or not probe.comparator:
        result.verdict = "empty"
        result.reason = "no threshold parsed"
        return result

    keep = [
        r
        for r, n in numeric
        if (n > probe.threshold if probe.comparator == ">" else n < probe.threshold)
    ]
    if probe.aggregate == "count":
        column = vocab.column_display.get(probe.column.resolved, probe.column.resolved)
        first = rows[0]
        result.verdict = "confident"
        result.facts = [
            ExactFact(
                entity=f"rows with {column} {probe.comparator} {probe.threshold:g}",
                column="count",
                value=str(len(keep)),
                source=first["source"],
                page=int(first["page"] or 0),
                table_title=first["table_title"],
                table_id=first["table_id"],
                kind="count",
            )
        ]
        return result

    if not keep:
        result.verdict = "empty"
        result.reason = "no rows past the threshold"
        return result
    if len(keep) > config.PARALLEL_SQL_MAX_FILTER_ROWS:
        result.verdict = "ambiguous"
        result.reason = f"{len(keep)} rows past the threshold, over the cap"
        return result

    result.verdict = "confident"
    result.facts = [_fact(r, kind="filter") for r in keep]
    return result


def execute(probe: Probe) -> SqlResult:
    """Run the probe's query. Safe to call on a worker thread.

    Takes ``user_id`` from the probe rather than the request-scoped contextvar:
    contextvars do not cross a thread boundary, and a scoping rule that silently
    evaporates on another thread is worse than no rule. Every read below has the
    owner in its WHERE clause.

    Never raises. A SQL failure must degrade to "the passages answer alone",
    which is exactly the pre-Addendum behaviour -- not to a failed request.
    """
    result = SqlResult(started_at=time.perf_counter())
    try:
        if not probe.fire:
            result.reason = probe.reason
            return result
        if probe.aggregate:
            inner = _aggregate(probe)
        else:
            inner = _single_cell(probe)
            # A cell-shaped question that found nothing may still be a
            # superlative-free aggregate ("total X"); no fallback is attempted,
            # because guessing a second interpretation after the first missed is
            # exactly how a wrong confident answer gets built.
        result.verdict = inner.verdict
        result.facts = inner.facts
        result.reason = inner.reason
        result.rows_returned = inner.rows_returned
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SQL lookup failed, falling back to retrieval only: %s", exc)
        result.verdict = "skipped"
        result.reason = f"error: {exc}"
        result.facts = []
    finally:
        result.finished_at = time.perf_counter()
    return result


# ---------------------------------------------------------------------------
# Rendering into the prompt
# ---------------------------------------------------------------------------

FACT_INSTRUCTION = (
    "For any numeric or exact value below, use the EXACT FACT verbatim -- never "
    "recompute, round, reformat, or paraphrase it. Use the context passages for "
    "explanation and surrounding detail, and answer the question fully rather "
    "than returning the bare value."
)


def render_facts(result: SqlResult) -> str:
    """The EXACT FACT block, or "" when there is nothing confident to show.

    Rendered as its own labelled block ahead of the passages so the model can see
    which of the two is authoritative for a value. It is appended to the text the
    grounding verifier reads as well, because the fact IS supported -- by the
    stored cell -- and a verifier that could not see it would strip the exact
    value back out of the answer it was added to.
    """
    if not result.confident:
        return ""
    lines = [FACT_INSTRUCTION, ""]
    for fact in result.facts:
        lines.append(f"EXACT FACT (from table, authoritative): {fact.line()}")
        lines.append(fact.citation())
    return "\n".join(lines)
