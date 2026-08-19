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

# Marker suffix for the synthetic "row count" cell emitted once per table row
# (see ``cells_from_tables``). Its column_norm is the entity column's own
# normalised name plus this marker ("vendor entity_type") -- kept OUT of the
# general column vocabulary and resolved through its own dedicated fuzzy pass
# (``vocab.entity_type_columns``, built in ``_build_vocabulary``) instead of
# ``_resolve``'s ordinary column search, so "how many vendors" is never
# compared against real attribute columns that happen to share a word with
# "vendor" ("Vendor Type", say) -- naming the KIND of row and naming an
# ATTRIBUTE of it are different questions, resolved separately.
_ENTITY_TYPE_MARKER = " entity_type"


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
        table_id = getattr(table, "table_id", "")
        entity_type_column_norm = normalise(entity_column) + _ENTITY_TYPE_MARKER

        alias_index = None
        if len(headers) > 2:
            col0 = [str(r[0]) if r else "" for r in rows]
            col1 = [str(r[1]) if len(r) > 1 else "" for r in rows]
            # A genuine second row label ("Name" alongside "Employee ID") is
            # near-unique per row, the same as any row label must be to
            # identify one row. A CATEGORY column ("Server Rack", "Laptop")
            # can look non-ID-shaped too but is deliberately LOW cardinality
            # -- many rows share one value -- and treating it as a second
            # entity means every row is stored under both its real id AND
            # its category, so a SUM/COUNT/filter over the table double-
            # counts every row whose category has more than one member.
            # Found by testing SUM against a synthetic asset register whose
            # second column was "Category": total came back exactly 2x the
            # correct sum, traced to this exact path.
            nonblank = [v for v in col1 if v.strip()]
            looks_unique = bool(nonblank) and len(set(nonblank)) / len(nonblank) >= 0.9
            if _looks_like_id_values(col0) and not _looks_like_id_values(col1) and looks_unique:
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

            # One synthetic cell per row, under the reserved entity-type
            # column_norm (the entity column's own name + marker), value =
            # the row's own entity label. Counting these rows is how "how
            # many vendors are there" answers as a plain COUNT(*) over the
            # table -- see ``_detect_aggregate``'s bare-count branch --
            # without needing a numeric column to exist at all. Stored as a
            # real cell so ownership/counting reuse the same table_cells
            # machinery every other column does; resolved through its own
            # fuzzy pass (``vocab.entity_type_columns``), not the general
            # column search, per the module constant's docstring above.
            if entity_norm:
                out.append(
                    {
                        "table_id": table_id,
                        "source": getattr(table, "source", ""),
                        "table_title": title,
                        "page": int(page or 0),
                        "row_index": index,
                        "row_entity": entity,
                        "row_entity_norm": entity_norm,
                        "column_name": entity_column,
                        "column_norm": entity_type_column_norm,
                        "value": entity,
                    }
                )
                if len(out) >= limit:
                    logger.warning(
                        "table cell cap %d reached; later cells not stored", limit
                    )
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
    # Bare-count support: entity noun ("vendor") -> the reserved column_norm
    # that counts its table's rows, and separately -> the table_id it counts.
    # Kept apart from ``columns`` (see ``_ENTITY_TYPE_MARKER``'s docstring).
    # Built from the SAME ``table_vocabulary`` rows as everything else -- no
    # second query.
    entity_type_columns: dict[str, str] = field(default_factory=dict)
    entity_type_display: dict[str, str] = field(default_factory=dict)
    entity_type_tables: dict[str, str] = field(default_factory=dict)
    # Multi-condition filter support: column_norm -> {value_norm: display}.
    # Lets "Under Repair" / "Pune" resolve to the COLUMN each is a value OF
    # (Status, Site) before a filter can be built spanning more than one
    # condition. A second query (``db.column_values``) because it is a
    # different shape of read than row/column vocabulary -- distinct VALUES,
    # not distinct labels -- not because it is a different cache lifetime;
    # it is invalidated and rebuilt in lockstep with everything else here.
    column_values: dict[str, dict[str, str]] = field(default_factory=dict)
    # column_norm -> table_id, for resolving which table a matched VALUE
    # belongs to when building a multi-condition filter.
    column_value_tables: dict[str, str] = field(default_factory=dict)

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
    entity_type_columns: dict[str, str] = {}
    entity_type_display: dict[str, str] = {}
    entity_type_tables: dict[str, str] = {}

    for row in rows:
        table_id = row["table_id"]
        column_norm = row["column_norm"]
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
        # Every row still counts toward this table's entity list and the
        # TableInfo row count, regardless of which cell (ordinary or
        # entity-type) carried the row_entity_norm -- both point at the same
        # underlying row. Only the COLUMN side excludes the entity-type
        # marker, which is a synthetic bookkeeping column, not something a
        # question should ever fuzzy-match as an attribute name.
        entry["entities"][row["row_entity_norm"]] = row["row_entity"]
        entity_display.setdefault(row["row_entity_norm"], row["row_entity"])

        if column_norm.endswith(_ENTITY_TYPE_MARKER):
            noun = column_norm[: -len(_ENTITY_TYPE_MARKER)]
            entity_type_columns.setdefault(noun, column_norm)
            entity_type_display.setdefault(noun, row["column_name"])
            entity_type_tables.setdefault(noun, table_id)
            continue

        entry["columns"][column_norm] = row["column_name"]
        column_display.setdefault(column_norm, row["column_name"])
        column_tables.setdefault(column_norm, set()).add(table_id)

    value_rows = db.column_values(user_id)
    column_values: dict[str, dict[str, str]] = {}
    column_value_tables: dict[str, str] = {}
    for row in value_rows:
        column_norm = row["column_norm"]
        if column_norm.endswith(_ENTITY_TYPE_MARKER):
            continue  # the row label itself, not a filterable attribute value
        value = row["value"]
        value_norm = value.casefold()
        column_values.setdefault(column_norm, {})[value_norm] = value
        # Deliberately last-writer-wins, same as column_tables' "spans N
        # tables" check catches downstream: a column name reused across two
        # different tables is a genuine ambiguity the aggregate/filter path
        # already refuses, not something this cache needs to pre-resolve.
        column_value_tables[column_norm] = row["table_id"]

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
        entity_type_columns=entity_type_columns,
        entity_type_display=entity_type_display,
        entity_type_tables=entity_type_tables,
        column_values=column_values,
        column_value_tables=column_value_tables,
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


def _resolve_extra_columns(
    candidates: list[Span],
    claimed: list[Span],
    seen: set[str],
    choices: tuple[str, ...],
    display: dict,
) -> list[TermMatch]:
    """Other columns EXACTLY named alongside the primary one, for one entity.

    ``_resolve`` stops at the first exact match it finds (longest-first), which
    is right for a single-column question but means "location and salary
    band" only ever resolves whichever phrase happened to be checked first --
    observed: always "salary band", never "location". Exact-only and
    non-overlapping-span-only, deliberately more conservative than the
    primary resolution's fuzzy fallback: a false EXTRA column costs a second,
    wrong authoritative-looking fact, not just a missed opportunity.
    """
    available = set(choices)
    found: list[TermMatch] = []
    for span in candidates:
        if any(span.overlaps(c) for c in claimed):
            continue
        for variant in _variants(span.text):
            if variant in available and variant not in seen:
                found.append(
                    TermMatch(
                        query_text=span.text,
                        resolved=variant,
                        display=display.get(variant, variant),
                        score=100.0,
                        span=span,
                    )
                )
                seen.add(variant)
                claimed.append(span)
                break
    return found


_VALUE_FUZZY_FLOOR = 88  # deliberately stricter than PARALLEL_SQL_FIRE_THRESHOLD (78):
# a value match seeds a WHERE clause that silently narrows a result set --
# a loose match here does not fail loudly like a missed lookup would, it
# just quietly drops rows that should have matched or keeps ones that
# should not have. See ``_resolve_value_filters``.


def _resolve_value_filters(
    candidates: list[Span], claimed: list[Span], vocab: "Vocabulary"
) -> list[tuple[TermMatch, TermMatch]]:
    """Every (column, value) condition a question names, for a multi-condition
    filtered list -- "Server Racks at the Pune site" is Category='Server Rack'
    AND Site='Pune'.

    Each span is checked against every column's distinct-value set
    (``vocab.column_values``, built alongside the rest of the vocabulary in
    ``_build_vocabulary``) rather than the general column vocabulary, because
    the question here names a VALUE ("Pune"), not an attribute NAME ("Site").
    Exact match first, then fuzzy at a stricter floor than everything else in
    this module (see ``_VALUE_FUZZY_FLOOR``) -- a wrong entity/column match
    only costs a missed opportunity (Decision 2 discards it), but a wrong
    VALUE match silently changes which rows a WHERE clause keeps.

    Longest spans win first and claim their words, so a four-word span that
    matches is preferred over the two-word span inside it matching a second,
    unrelated value coincidentally -- the same longest-first discipline
    ``_resolve`` uses for columns/entities.
    """
    found: list[tuple[TermMatch, TermMatch]] = []
    local_claimed = list(claimed)
    seen_columns: set[str] = set()

    for span in candidates:
        if any(span.overlaps(c) for c in local_claimed):
            continue
        text = span.text
        # A short span ("are", "for", "the") scores deceptively high against
        # an unrelated long value purely from WRatio's partial-ratio
        # component -- the same failure ``_resolve`` already guards against
        # for columns/entities, via the "combine with plain ratio, lower
        # wins" rule below. Filler words never name a value on their own, so
        # they are excluded outright rather than merely down-weighted: a
        # value match is a WHERE-clause condition, and ``_VALUE_FUZZY_FLOOR``
        # is only a safe floor once the candidate is a real word.
        if len(text) < 4 or text in _FILLER_WORDS:
            continue
        best_column = ""
        best_value_match: TermMatch | None = None
        for column_norm, values in vocab.column_values.items():
            if column_norm in seen_columns:
                continue  # one condition per column -- a second value for the
                # same column would be a contradiction (Status='X' AND
                # Status='Y'), never a second real condition.
            if text in values:
                best_column, best_value_match = column_norm, TermMatch(
                    query_text=text,
                    resolved=text,
                    display=values[text],
                    score=100.0,
                    span=span,
                )
                break  # exact beats fuzzy; stop at the first exact hit
            hit = process.extractOne(text, list(values), scorer=fuzz.WRatio)
            if not hit:
                continue
            score = float(hit[1])
            if len(hit[0]) <= 6:
                score = min(score, float(fuzz.ratio(text, hit[0])))
            if score >= _VALUE_FUZZY_FLOOR:
                if best_value_match is None or score > best_value_match.score:
                    best_column, best_value_match = column_norm, TermMatch(
                        query_text=text,
                        resolved=hit[0],
                        display=values[hit[0]],
                        score=score,
                        span=span,
                    )
        if best_value_match is not None:
            column_match = TermMatch(
                query_text=text,
                resolved=best_column,
                display=vocab.column_display.get(best_column, best_column),
                score=100.0,
                span=span,
            )
            found.append((column_match, best_value_match))
            seen_columns.add(best_column)
            local_claimed.append(span)

    return found


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
    # A compound question ("location AND salary band of Employee 100") names
    # more than one column for the same entity. ``column`` stays whichever one
    # `_resolve` found first, for backward-compatible diagnostics; these ride
    # alongside it so a second attribute is not silently dropped.
    extra_columns: list[TermMatch] = field(default_factory=list)
    multi_answer_cue: str = ""
    numeric_cue: str = ""
    aggregate: str = ""          # "", "max", "min", "count", "sum", "filter", "row_count"
    threshold: float | None = None
    comparator: str = ""         # ">" or "<"
    vocabulary_size: int = 0
    # Bare-count support ("how many vendors are there"): which entity noun
    # resolved, and which table it counts. Separate from ``column`` because
    # it is resolved against ``vocab.entity_type_columns``, not the ordinary
    # column vocabulary -- see ``_ENTITY_TYPE_MARKER``.
    entity_type: TermMatch = field(default_factory=TermMatch)
    entity_type_table: str = ""
    # Multi-condition filtered list ("Server Racks at the Pune site"): every
    # (column, value) pair the question named, resolved against
    # ``vocab.column_values``. A single-condition filter ("Category" ==
    # "Server Rack") is just a value_filters list of length 1 -- there is no
    # separate single-condition code path.
    value_filters: list[tuple[TermMatch, TermMatch]] = field(default_factory=list)
    # "How many assets are Under Repair" wants a NUMBER, not the row list
    # ("List the Under Repair assets" wants the list). Same filter query
    # either way; only what ``_multi_filter`` renders as a fact differs.
    filter_count_only: bool = False


# "best"/"worst" are deliberately absent: they are subjective, not numeric, and
# "who is the best engineer" is not a MAX over a column.
_SUPERLATIVE_MAX = ("highest", "greatest", "largest", "most", "top", "maximum", "max")
_SUPERLATIVE_MIN = ("lowest", "smallest", "least", "minimum", "min")
_ABOVE = ("above", "over", "more than", "greater than", "at least", "exceeds")
_BELOW = ("below", "under", "less than", "fewer than", "at most")
# "most" is deliberately absent here too: it is already a MAX cue above, and
# "the most expensive asset" must resolve to a single winning row, not a sum.
_SUM_CUES = (
    "total value", "total cost", "total amount", "sum of", "combined",
    "aggregate", "grand total", "total worth",
)
# Bare row-count over a whole table, naming the KIND of row rather than a
# column: "how many vendors", "number of trainings", "count of assets",
# "total number of". Distinct from ``NUMERIC_CUES``'s "how many"/"total",
# which fire Decision 1 speculatively -- this decides WHICH aggregate kind,
# once something has already fired.
_COUNT_CUES = ("how many", "number of", "count of", "total number")
_LIST_CUES = ("list all", "which ones", "show all", "every", "all the")

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

    sum_cue = _has_cue(question, _SUM_CUES)
    if sum_cue and not _has_cue(named, (sum_cue,)):
        return "sum", "", None

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

    Gated on ``PARALLEL_SQL_LOOKUP_ENABLED``: this is the entry point for
    actually EXECUTING a speculative SQL lookup, and that costs a worker-thread
    submission per call even when nothing fires. `table-router` branch:
    :func:`classify_table_relatedness` needs the same resolution this function
    does, but as a pure routing SIGNAL that must work regardless of whether SQL
    lookup itself is enabled -- so it calls :func:`_probe_uncached` directly,
    skipping this gate.
    """
    if not config.PARALLEL_SQL_LOOKUP_ENABLED:
        probe = Probe(user_id=user_id, question=question)
        probe.reason = "flag off"
        return probe
    return _probe_uncached(user_id, question)


def _probe_uncached(user_id: str, question: str) -> Probe:
    """The actual Decision 1 resolution, without the feature-flag gate.

    Split out of :func:`prepare` so :func:`classify_table_relatedness` (the
    `table-router` branch's classifier) can reuse the exact same fuzzy
    entity/column resolution against the user's table vocabulary, independent
    of whether SQL lookup EXECUTION is enabled -- routing to ColPali and
    running the SQL fast-path are separate features with separate on/off
    switches.
    """
    probe = Probe(user_id=user_id, question=question)
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

    claimed = [s for s in (probe.column.span, probe.entity.span) if s]
    seen = {probe.column.resolved} if probe.column.resolved else set()
    probe.extra_columns = _resolve_extra_columns(
        candidates, claimed, seen, vocab.columns, vocab.column_display
    )

    probe.multi_answer_cue = _has_cue(question, MULTI_ANSWER_CUES)
    probe.numeric_cue = _has_cue(question, NUMERIC_CUES)
    # After resolution, not before: both aggregate suppressions need to know what
    # the entity and the column actually resolved to.
    probe.aggregate, probe.comparator, probe.threshold = _detect_aggregate(
        question, probe.column, probe.entity
    )

    # Bare row-count / filtered-list over a whole table: "how many vendors",
    # "list all Server Racks at the Pune site". Gated on TWO conditions, both
    # required, not attempted just because a table exists:
    #
    # 1. The column-bound aggregate above came back empty -- a question that
    #    resolved a real numeric column (MAX/MIN/SUM/threshold) already has
    #    its answer.
    # 2. A trustworthy single-cell lookup did NOT already resolve as an
    #    unambiguous ONE-ANSWER question -- same suppression rule
    #    ``_detect_aggregate`` already applies for MAX/MIN/SUM ("a question
    #    that names a specific ROW is a lookup, not an aggregate"): "What
    #    tier is Northwind Logistics?" resolves entity+column cleanly with NO
    #    multi-answer cue and must stay a single-cell answer. "List all the
    #    fault codes ... and what each one means" also resolves entity+
    #    column, but its "all" cue means it wants every row, not the one
    #    ``_single_cell`` would have matched -- so a multi-answer/list/count
    #    cue overrides the single-cell shortcut here exactly the way it
    #    already overrides Decision 2's trust in ``_single_cell`` itself.
    # 3. The question actually carries a count/sum/list CUE -- without this,
    #    every ordinary lookup question would probe every column's value
    #    vocabulary for no reason, and "Northwind Logistics" would spuriously
    #    fuzzy-match some unrelated column's stored value at the 88 floor.
    trust = config.PARALLEL_SQL_TRUST_THRESHOLD
    count_cue = _has_cue(question, _COUNT_CUES)
    list_cue = _has_cue(question, _LIST_CUES)
    wants_many = bool(count_cue or list_cue or probe.multi_answer_cue)
    single_cell_would_resolve = (
        probe.entity.score >= trust and probe.column.score >= trust and not wants_many
    )
    if not probe.aggregate and not single_cell_would_resolve and wants_many:
        probe.entity_type = _resolve(
            candidates, tuple(vocab.entity_type_columns), vocab.entity_type_display
        )
        value_claimed = claimed + (
            [probe.entity_type.span] if probe.entity_type.span else []
        )
        probe.value_filters = _resolve_value_filters(candidates, value_claimed, vocab)
        if probe.entity_type.resolved:
            probe.entity_type_table = vocab.entity_type_tables.get(
                probe.entity_type.resolved, ""
            )
        if probe.value_filters:
            probe.aggregate = "filter"
            probe.filter_count_only = count_cue and not list_cue
        elif probe.entity_type.resolved and count_cue:
            probe.aggregate = "row_count"
        elif probe.entity_type.resolved and list_cue:
            probe.aggregate = "filter"  # list-all with no condition: every row

    floor = config.PARALLEL_SQL_FIRE_THRESHOLD
    reasons = []
    if probe.column.score >= floor:
        reasons.append(f"column~{probe.column.resolved}({probe.column.score:.0f})")
    for extra in probe.extra_columns:
        reasons.append(f"column~{extra.resolved}(100)")
    if probe.entity.score >= floor:
        reasons.append(f"entity~{probe.entity.resolved}({probe.entity.score:.0f})")
    if probe.entity_type.score >= floor:
        reasons.append(f"entity_type~{probe.entity_type.resolved}({probe.entity_type.score:.0f})")
    for col_match, val_match in probe.value_filters:
        reasons.append(f"filter~{col_match.resolved}={val_match.resolved}({val_match.score:.0f})")
    if probe.numeric_cue:
        reasons.append(f"cue:{probe.numeric_cue}")

    probe.fire = bool(reasons)
    probe.reason = ", ".join(reasons) if reasons else "no column, entity, or numeric cue"
    return probe


@dataclass(frozen=True)
class TableRelatedness:
    """Verdict for the `table-router` branch's LangGraph classifier node.

    ``is_table_related`` is the routing bit; ``reason`` and ``probe`` are
    carried through so a caller (the router node, or a log line) can explain
    the classification without re-running resolution.
    """

    is_table_related: bool
    reason: str
    probe: Probe


def classify_table_relatedness(user_id: str, question: str) -> TableRelatedness:
    """Does ``question`` reference a specific row/cell/entity known to live in
    a table, or ask about content/values inside one?

    `table-router` branch: the CLASSIFIER node's decision signal. Reuses the
    same fuzzy entity/column resolution as Decision 1 (:func:`_probe_uncached`)
    rather than a second, drifting word-list -- "is this question about my
    tables" and "should SQL fire" both start from the same underlying
    question (does a term in it resolve against this user's column/entity
    vocabulary?). Distinct from Decision 1 in two respects:

    1. It runs regardless of ``PARALLEL_SQL_LOOKUP_ENABLED``, since routing to
       ColPali is a separate concern from executing the SQL fast-path.
    2. It gates on Decision 2's STRICT trust floor
       (``PARALLEL_SQL_TRUST_THRESHOLD``, 85 by default), not Decision 1's
       permissive fire floor (78, "fire on any hint"). Decision 1 can afford
       to be permissive because Decision 2 vetoes a bad match before it is
       ever stated as fact -- the router has no such second gate, and a
       weak/ambiguous fuzzy hit routed to the wrong backend is not a no-op
       the way a discarded SQL result is.
    3. It re-checks a strong match with a SHORT query span against plain
       edit-distance, via :func:`_confident_match` below. Measured: "What is
       the process for requesting remote work?" fuzzy-matches the word
       "work" against the stored fault-code entity "network link lost" at
       WRatio 90 -- over even the strict floor -- because WRatio's
       partial-ratio component rewards "work" appearing as a near-substring
       fragment of "network". ``_resolve``'s own docstring already documents
       this exact failure mode for a SHORT CANDIDATE ("Bo" inside "above")
       and guards it there by combining WRatio with plain ``fuzz.ratio``; the
       guard is keyed on ``len(hit[0])`` (the candidate's length) so it does
       not cover the reverse case -- a short QUERY span against a long
       candidate -- which is exactly this failure. That asymmetry is not
       fixed in ``_resolve`` itself: doing so would also change Decision 1's
       fire signal for the already-shipped SQL aggregation feature, which is
       out of scope here. ``_confident_match`` applies the identical
       combination rule, scoped to this classifier only.

    A user with no stored tables (no ingested document produced one, or
    Addendum 2 has never populated ``table_cells`` for them) can never be
    classified table-related -- there is no vocabulary to match against, so
    every question falls through to NORMAL_ROUTE_BACKEND.
    """
    probe = _probe_uncached(user_id, question)
    if probe.vocabulary_size == 0:
        return TableRelatedness(False, probe.reason or "no stored tables", probe)

    trust = config.PARALLEL_SQL_TRUST_THRESHOLD
    is_related = (
        _confident_match(probe.column, trust)
        or _confident_match(probe.entity, trust)
        or _confident_match(probe.entity_type, trust)
        or bool(probe.value_filters)
        or bool(probe.aggregate)
    )
    return TableRelatedness(is_related, probe.reason, probe)


def _confident_match(match: TermMatch, trust: float) -> bool:
    """Is ``match`` a genuinely confident resolution, not a short-query-span
    coincidence? See :func:`classify_table_relatedness` point 3 above.
    """
    if match.score < trust:
        return False
    if len(match.query_text) <= 6:
        return fuzz.ratio(match.query_text, match.resolved) >= trust
    return True


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
    kind: str = "cell"  # "cell", "max", "min", "count", "sum", "row_count", "filter"

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
    """The lookup path: one resolved entity, one resolved column, one value.

    ``probe.extra_columns`` (a compound question -- "location AND salary
    band") ride along AFTER the primary column's own Decision-2 verdict is
    settled: adding a second attribute must never turn a solid single-column
    answer into "ambiguous" just because the second one did not resolve as
    cleanly. Each extra fact is included only if it independently lands on
    exactly one row.
    """
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
        for extra in probe.extra_columns:
            extra_rows = db.lookup_cells(
                probe.user_id, probe.entity.resolved, extra.resolved
            )
            result.rows_returned += len(extra_rows)
            if len(extra_rows) == 1:
                result.facts.append(_fact(extra_rows[0]))
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

    if probe.aggregate == "sum":
        # Every discarded (non-numeric) row is REPORTED, not silently
        # dropped: a sum computed over 178 of 180 rows while the answer
        # implies completeness is exactly the dishonest-confidence failure
        # this whole module exists to avoid. ``rows_returned`` already holds
        # the parsed count; ``rows` (pre-filter) holds the true total.
        total = sum(n for _, n in numeric)
        discarded = len(rows) - len(numeric)
        column = vocab.column_display.get(probe.column.resolved, probe.column.resolved)
        first = rows[0]
        coverage = f"{len(numeric)} of {len(rows)} rows"
        entity_label = f"total {column}" if discarded == 0 else (
            f"total {column} ({coverage}; {discarded} row(s) had no numeric "
            f"value and were excluded)"
        )
        result.verdict = "confident"
        result.facts = [
            ExactFact(
                entity=entity_label,
                column="sum",
                value=f"{total:g}",
                source=first["source"],
                page=int(first["page"] or 0),
                table_title=first["table_title"],
                table_id=first["table_id"],
                kind="sum",
            )
        ]
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


def _row_count(probe: Probe) -> SqlResult:
    """Bare COUNT(*) over a whole table -- "how many vendors are there",
    with no column and no threshold at all. The entity-type resolution in
    ``prepare`` already picked the table; this just counts its distinct
    entities, which ``TableInfo.entities`` already holds in memory -- no
    query needed for the number itself, only for the citation.
    """
    result = SqlResult()
    trust = config.PARALLEL_SQL_TRUST_THRESHOLD

    if probe.entity_type.score < trust:
        result.verdict = "ambiguous"
        result.reason = f"entity-type match {probe.entity_type.score:.0f} < {trust}"
        return result
    if not probe.entity_type_table:
        result.verdict = "empty"
        result.reason = "entity-type column resolved to no table"
        return result

    vocab = vocabulary(probe.user_id)
    info = vocab.tables.get(probe.entity_type_table)
    if info is None:
        result.verdict = "empty"
        result.reason = "table vanished between resolve and execute"
        return result

    result.rows_returned = len(info.entities)
    noun = vocab.entity_type_display.get(probe.entity_type.resolved, probe.entity_type.resolved)
    result.verdict = "confident"
    result.facts = [
        ExactFact(
            entity=f"count of {noun} rows",
            column="count",
            value=str(len(info.entities)),
            source=info.source,
            page=info.page,
            table_title=info.title,
            table_id=info.table_id,
            kind="row_count",
        )
    ]
    return result


def _multi_filter(probe: Probe) -> SqlResult:
    """Filtered list over one or more (column, value) conditions --
    "Server Racks at the Pune site" is Category='Server Rack' AND
    Site='Pune'; "list all data classification tiers" (no condition
    resolved, just a list-cue) is every row of the resolved entity type.

    Every condition must resolve to the SAME table -- a filter spanning two
    unrelated tables is the same "category error" ``_aggregate`` already
    refuses for MAX/MIN, just with more than one column involved.

    Returns the SQL rows themselves, verbatim, as facts -- the model is
    handed the finished list (see ``render_facts``'s instruction not to
    recount or extend it), which is what keeps a fabricated ID like
    "AST-2188" from ever entering the answer: it was never generated, it was
    read directly out of ``table_cells``.
    """
    result = SqlResult()
    table_id = probe.entity_type_table or ""
    if probe.value_filters:
        owners = {
            vocabulary(probe.user_id).column_value_tables.get(col.resolved, "")
            for col, _ in probe.value_filters
        }
        owners.discard("")
        if len(owners) > 1:
            result.verdict = "ambiguous"
            result.reason = f"filter conditions span {len(owners)} tables"
            return result
        if owners:
            (only_owner,) = owners
            if table_id and table_id != only_owner:
                result.verdict = "ambiguous"
                result.reason = "entity-type table and filter-column table disagree"
                return result
            table_id = only_owner

    if not table_id:
        result.verdict = "empty"
        result.reason = "no table resolved for filter"
        return result

    vocab = vocabulary(probe.user_id)
    info = vocab.tables.get(table_id)
    if info is None:
        result.verdict = "empty"
        result.reason = "table vanished between resolve and execute"
        return result

    if not probe.value_filters:
        # No condition at all: "list all data classification tiers" names
        # only the entity type. Every row in the table is the answer. Read
        # through the reserved entity-type column rather than an arbitrary
        # attribute column -- it exists on every row by construction (see
        # ``cells_from_tables``), so this never depends on which attribute
        # columns happen to be non-empty.
        entity_type_norm = next(
            (k for k, v in vocab.entity_type_tables.items() if v == table_id), ""
        )
        all_cells = db.column_cells(
            probe.user_id,
            vocab.entity_type_columns.get(entity_type_norm, ""),
            table_id=table_id,
        )
        matching_indices = {int(r["row_index"]) for r in all_cells}
    else:
        matching_indices = None
        for col, val in probe.value_filters:
            rows_for_condition = db.rows_matching(
                probe.user_id, table_id, col.resolved, val.resolved
            )
            matching_indices = (
                rows_for_condition
                if matching_indices is None
                else matching_indices & rows_for_condition
            )
            if not matching_indices:
                break

    if not matching_indices:
        if probe.filter_count_only:
            # Zero IS the exact answer to "how many X are Y" -- not a miss.
            result.verdict = "confident"
            result.facts = [_zero_count_fact(probe, vocab, table_id)]
            return result
        result.verdict = "empty"
        result.reason = "no rows matched every condition"
        return result

    if probe.filter_count_only:
        # A COUNT never needs the row cap: the row cap protects against
        # dumping an unbounded LIST of rows into the prompt, and a bare
        # number costs nothing extra to state whether it is 2 or 200.
        first_id = next(iter(db.rows_by_index(probe.user_id, table_id, {min(matching_indices)})), None)
        result.rows_returned = len(matching_indices)
        result.verdict = "confident"
        condition = _filter_condition_label(probe, vocab)
        result.facts = [
            ExactFact(
                entity=f"count of rows where {condition}",
                column="count",
                value=str(len(matching_indices)),
                source=first_id["source"] if first_id else "",
                page=int(first_id["page"] or 0) if first_id else 0,
                table_title=first_id["table_title"] if first_id else "",
                table_id=table_id,
                kind="row_count",
            )
        ]
        return result

    if len(matching_indices) > config.PARALLEL_SQL_MAX_FILTER_ROWS:
        result.verdict = "ambiguous"
        result.reason = f"{len(matching_indices)} rows matched, over the cap"
        return result

    cells = db.rows_by_index(probe.user_id, table_id, matching_indices)
    result.rows_returned = len(cells)
    if not cells:
        result.verdict = "empty"
        result.reason = "matched rows had no stored cells"
        return result

    # One fact per matched ROW (its own entity label), not per cell -- the
    # brief's acceptance check wants exactly the ID list, not every
    # attribute of every matched row repeated back.
    by_row: dict[int, dict] = {}
    for cell in cells:
        by_row.setdefault(int(cell["row_index"]), cell)
    result.verdict = "confident"
    result.facts = [_fact(row, kind="filter") for row in by_row.values()]
    return result


def _filter_condition_label(probe: "Probe", vocab: "Vocabulary") -> str:
    if not probe.value_filters:
        return "no condition"
    return " and ".join(
        f"{vocab.column_display.get(col.resolved, col.resolved)} = {val.display}"
        for col, val in probe.value_filters
    )


def _zero_count_fact(probe: "Probe", vocab: "Vocabulary", table_id: str) -> ExactFact:
    info = vocab.tables.get(table_id)
    return ExactFact(
        entity=f"count of rows where {_filter_condition_label(probe, vocab)}",
        column="count",
        value="0",
        source=info.source if info else "",
        page=info.page if info else 0,
        table_title=info.title if info else "",
        table_id=table_id,
        kind="row_count",
    )


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
        # ``row_count`` and the multi-condition ``filter`` (identified by
        # ``entity_type``/``value_filters`` being set -- see ``prepare``,
        # where they are only attempted once the column-bound aggregate
        # below has already come back empty) are resolved via the
        # entity-type table rather than one resolved numeric column, so they
        # dispatch to their own functions rather than through ``_aggregate``.
        if probe.aggregate == "row_count":
            inner = _row_count(probe)
        elif probe.aggregate == "filter" and (probe.entity_type.resolved or probe.value_filters):
            inner = _multi_filter(probe)
        elif probe.aggregate:
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
    "than returning the bare value.\n"
    "If more than one EXACT FACT is listed below, together they ARE the "
    "complete answer set -- state exactly those items and nothing else. Do "
    "NOT add, drop, recount, re-derive, or estimate any item, and do NOT "
    "list example rows of your own; an item not listed below was not found, "
    "and inventing one is worse than a shorter answer."
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
