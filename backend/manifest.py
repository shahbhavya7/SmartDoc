"""V3.3 Layer C: a per-document manifest, and the routing it makes possible.

The problem
-----------
An enumeration question -- "all types of leave", "every approval", "all required
documents" -- is answered today by making retrieval *wide* and hoping top-k caught
every item. When it does not, the failure is **silent**: the answer lists three of
seven, reads as finished, and nothing in the system knows an item is missing.
Every other guard is looking elsewhere. Grounding verification asks "is each claim
supported?", not "is each item present?" -- and three correct items out of seven
pass that check perfectly.

A manifest turns "how many are there?" from a hope into a fact. After all of a
document's chunks exist, everything it *exactly* enumerates is written to SQLite:

* **sections** -- the V3.1 heading tree,
* **table rows** -- the V3.2 structured tables, which already hold every row label,
* **list items** -- bullet and numbered leads in the chunk text,

alongside the document's aggregated **topics** and **entities** from Layer B.

The structural half needs no LLM, so it is free, exact and reproducible. That
matters: the count is the thing being trusted, so it must not be a model's opinion.

What it drives
--------------
1. **Enumeration routing.** Read the authoritative list first, then fetch the chunk
   for each listed item by metadata -- so an item that would have ranked below the
   cut cannot be dropped. Then check the finished answer against the count and
   regenerate once naming what it missed.
2. **Heading filtering.** When a question names a section that exists, narrow the
   candidate pool by ``heading_path`` before searching.

Why SQLite
----------
Chroma stores scalars, so a list of seven leave types cannot live there. It is also
the wrong shape: "every item in this group" is a structural query, which is what a
relational store is for. Chroma keeps the vectors; the manifest keeps the counts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import backend.config as config
from backend import db
from backend.chunk_schema import split_list

# A bullet or numbered lead-in. Deliberately narrow: it must look like a list item,
# not merely a sentence that happens to start with a number.
_LIST_LEAD = re.compile(
    r"^\s*(?:[-•*+]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+(?P<body>\S.{2,120}?)\s*$",
    re.IGNORECASE,
)

# "Label: description" -- "Sick leave: ten days per year" enumerates *sick leave*.
_LABELLED = re.compile(r"^(?P<label>[^:–—-]{2,60})\s*[:–—-]\s+\S")

# A leading section number, stripped so an item reads as a thing rather than an
# outline row.
_SECTION_NUMBER = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")

_STOPWORDS = {
    "a", "all", "an", "and", "any", "are", "as", "at", "available", "be", "by",
    "can", "complete", "different", "do", "does", "each", "every", "everything",
    "exist", "exists", "for", "from", "full", "get", "give", "has", "have", "how",
    "in", "is", "it", "kind", "kinds", "list", "many", "me", "my", "of", "on",
    "or", "our", "please", "provide", "show", "some", "tell", "that", "the",
    "there", "these", "they", "this", "those", "to", "type", "types", "us", "we",
    "what", "which", "with", "you", "your",
}

# An enumeration asks for a SET. Over-inclusive on purpose -- it is only a cheap
# gate, and ``plan_enumeration`` returns None when no authoritative list backs it.
_ENUMERATION = re.compile(
    r"\b(all|every|each|list|enumerate|complete list|full list|"
    r"types? of|kinds? of|what are the|which are the|how many)\b",
    re.IGNORECASE,
)

# "the eligibility section", "in the scope section", "under Sick Leave"
_SECTION_REFERENCE = re.compile(
    r"\b(?:the\s+)?(?P<name>[\w][\w '\-/&]{2,60}?)\s+section\b"
    r"|\bsection\s+(?:on\s+|about\s+|called\s+)?(?P<named>[\w][\w '\-/&]{2,60})",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop a leading section number."""
    return re.sub(r"\s+", " ", _SECTION_NUMBER.sub("", text or "")).strip().lower()


def _singular(word: str) -> str:
    """Crude singulariser -- enough for approvals/tiers/categories."""
    lowered = word.lower()
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith(("sses", "ches", "shes")):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def word_match(term: str, text: str) -> bool:
    """Whole-word (or whole-phrase) containment, not raw substring.

    ``"code" in "codex"`` is True as a substring and wrong as a match -- it picked
    a 16-item group out of an unrelated document. Every word of the term must
    appear as a word; a stem match is allowed only in the direction that adds a
    plural ("training" matches "trainings"), never the reverse.
    """
    haystack = normalise(text)
    if not haystack:
        return False
    return all(
        re.search(rf"\b{re.escape(word)}(?:s|es|ies)?\b", haystack)
        for word in normalise(term).split()
    )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass
class ManifestItem:
    """One thing a document exactly enumerates."""

    kind: str  # "section" | "table_row" | "list_item"
    item: str
    group_label: str = ""
    group_context: str = ""
    heading_path: str = ""
    page: int | None = None
    chunk_index: int | None = None
    table_id: str = ""

    def as_row(self) -> dict:
        return {
            "kind": self.kind,
            "group_label": self.group_label,
            "group_context": self.group_context,
            "item": self.item,
            "item_norm": normalise(self.item),
            "heading_path": self.heading_path,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "table_id": self.table_id,
        }


@dataclass
class Manifest:
    """Everything known structurally about one document."""

    source: str
    heading_tree: list[dict] = field(default_factory=list)
    items: list[ManifestItem] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    chunk_count: int = 0
    table_count: int = 0


def build_heading_tree(children: list) -> list[dict]:
    """The heading hierarchy, in reading order, with page spans and chunk counts.

    Derived from the finished chunks rather than re-parsed, so the tree can only
    describe headings that chunks were actually filed under. A tree naming a
    section with no retrievable chunk would be a manifest promising evidence the
    index cannot produce.
    """
    seen: dict[str, dict] = {}
    for child in children:
        metadata = child.metadata
        path = str(metadata.get("heading_path") or "").strip()
        if not path:
            continue
        page = int(metadata.get("page") or 0)
        node = seen.get(path)
        if node is None:
            parts = [p.strip() for p in path.split(">") if p.strip()]
            seen[path] = {
                "path": path,
                "title": parts[-1] if parts else path,
                "level": len(parts),
                "page_start": page,
                "page_end": int(metadata.get("page_end") or page),
                "chunks": 1,
            }
        else:
            node["page_end"] = max(node["page_end"], int(metadata.get("page_end") or page))
            node["chunks"] += 1
    return list(seen.values())


def _aggregate(children: list, key: str, limit: int = 40) -> list[str]:
    """Union of a Layer B list field across the document, most frequent first.

    This is the "aggregated entity/topic lists per document" half of the manifest:
    a document-level view of what it is about, which no single chunk carries.
    """
    counts: dict[str, int] = {}
    for child in children:
        for value in split_list(child.metadata.get(key, "")):
            counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [value for value, _ in ranked[:limit]]


def _section_items(tree: list[dict]) -> list[ManifestItem]:
    """Every heading becomes an item, grouped under its parent path."""
    items: list[ManifestItem] = []
    for node in tree:
        parts = [p.strip() for p in node["path"].split(">") if p.strip()]
        if len(parts) < 2:
            continue  # the document title is not one of its own items
        items.append(
            ManifestItem(
                kind="section",
                item=_SECTION_NUMBER.sub("", parts[-1]).strip() or parts[-1],
                group_label=" > ".join(parts[:-1]),
                heading_path=node["path"],
                page=node["page_start"],
            )
        )
    return items


def _table_items(tables: list) -> list[ManifestItem]:
    """Every table row label becomes an item, grouped by its table.

    V3.2 already extracted these exactly, headers included, so this is the one
    enumeration source that is guaranteed complete: the manifest knows a fault-code
    table has nine rows because nine rows were parsed.
    """
    items: list[ManifestItem] = []
    for table in tables:
        # The column headers are how a reader NAMES this group ("record types",
        # "fault codes"), which is often not what the enclosing section is called.
        context = " | ".join(c for c in table.headers if c)
        for row in table.rows:
            if not row:
                continue
            first = str(row[0] or "").replace("\n", " ").strip()
            if first:
                items.append(
                    ManifestItem(
                        kind="table_row",
                        item=first,
                        group_label=table.section or table.source,
                        group_context=context,
                        page=table.page_start,
                        table_id=table.table_id,
                    )
                )
    return items


def _list_items(children: list, max_per_chunk: int = 12) -> list[ManifestItem]:
    """Bullet and numbered items in chunk text, grouped by their section.

    Capped per chunk: a chunk that is one long bulleted list is a legitimate
    enumeration, but an over-firing regex would otherwise fill the manifest with
    sentence fragments and make every enumeration noisy.
    """
    items: list[ManifestItem] = []
    for child in children:
        metadata = child.metadata
        if metadata.get("table_id"):
            continue  # table rows are handled exactly, above
        found = 0
        for line in (child.page_content or "").split("\n"):
            match = _LIST_LEAD.match(line)
            if not match:
                continue
            body = match.group("body").strip()
            labelled = _LABELLED.match(body)
            text = (labelled.group("label") if labelled else body).strip(" .;:")
            if len(text) < 3 or len(text.split()) > 10:
                continue
            heading = str(metadata.get("heading_path") or "")
            items.append(
                ManifestItem(
                    kind="list_item",
                    item=text,
                    group_label=heading,
                    heading_path=heading,
                    page=int(metadata.get("page") or 0),
                    chunk_index=int(metadata.get("chunk_index") or 0),
                )
            )
            found += 1
            if found >= max_per_chunk:
                break
    return items


def build_manifest(parsed, children: list) -> Manifest:
    """Build a document's manifest from its parsed form and its finished chunks.

    Called after chunking and after Layer B, not during either: an item has to name
    the chunk that will answer for it, and chunk indices do not exist until the
    whole stream (prose and tables) has been merged and numbered.
    """
    tree = build_heading_tree(children)
    tables = list(getattr(parsed, "tables", []) or [])
    items = _section_items(tree) + _table_items(tables) + _list_items(children)

    # Dedupe on (kind, group, normalised item): the same bullet appears in two
    # overlapping chunks, and counting it twice makes "there are 7" wrong.
    unique: dict[tuple[str, str, str], ManifestItem] = {}
    for item in items:
        unique.setdefault((item.kind, item.group_label, normalise(item.item)), item)

    return Manifest(
        source=parsed.source,
        heading_tree=tree,
        items=list(unique.values()),
        topics=_aggregate(children, "topics"),
        entities=_aggregate(children, "entities"),
        chunk_count=len(children),
        table_count=len(tables),
    )


def store_manifest(user_id: str, document_id: str, built: Manifest) -> int:
    """Persist a manifest, replacing any previous one for this document."""
    if not user_id or not document_id:
        # An unscoped ingest (the evaluation harness) has no documents row to hang
        # a manifest off. Returning 0 rather than raising keeps that path working.
        return 0
    return db.replace_manifest(
        user_id=user_id,
        document_id=document_id,
        source=built.source,
        heading_tree_json=json.dumps(built.heading_tree, ensure_ascii=False),
        topics_json=json.dumps(built.topics, ensure_ascii=False),
        entities_json=json.dumps(built.entities, ensure_ascii=False),
        items=[item.as_row() for item in built.items],
        table_count=built.table_count,
        chunk_count=built.chunk_count,
    )


# ---------------------------------------------------------------------------
# Use 1: enumeration routing
# ---------------------------------------------------------------------------


def is_enumeration(question: str) -> bool:
    """Cheap gate: does this question ask for a SET rather than a value?"""
    return bool(_ENUMERATION.search(question or ""))


def enumeration_terms(question: str, limit: int = 8) -> list[str]:
    """Candidate search terms, MOST SPECIFIC FIRST.

    Adjacent content-word bigrams come before single words, because "fault code"
    identifies a group that "code" does not.
    """
    words = [
        w for w in re.findall(r"[A-Za-z][\w-]{2,}", question or "")
        if w.lower() not in _STOPWORDS
    ]
    bigrams = [f"{_singular(a)} {_singular(b)}" for a, b in zip(words, words[1:])]
    unigrams = [_singular(w) for w in words]
    seen: dict[str, None] = {}
    for term in bigrams + unigrams:
        if len(term) >= 3:
            seen.setdefault(term, None)
    return list(seen)[:limit]


def head_noun(question: str) -> str:
    """The first content word -- what the enumeration is actually about."""
    return next(
        (
            _singular(w)
            for w in re.findall(r"[A-Za-z][\w-]{2,}", question or "")
            if w.lower() not in _STOPWORDS
        ),
        "",
    )


@dataclass
class EnumerationPlan:
    """What the manifest says a complete answer must cover."""

    term: str
    items: list[dict]
    truncated_from: int = 0

    @property
    def expected(self) -> int:
        return len(self.items)

    @property
    def labels(self) -> list[str]:
        return [str(row["item"]) for row in self.items]

    def missing_from(self, answer: str) -> list[str]:
        """Expected labels the answer does not mention.

        Two passes: whole-label containment, then all significant tokens present.
        Short tokens are kept in the second pass -- dropping them made "Item c"
        count as covered because "item" appeared somewhere.

        The bias is deliberate. A false "missing" costs one regeneration that can
        only improve the answer or be discarded; a false "covered" leaves the
        silent incompleteness this whole layer exists to catch.
        """
        haystack = normalise(answer)
        filler = {"and", "or", "of", "the", "a", "an", "for", "to", "in", "by"}
        missing: list[str] = []
        for label in self.labels:
            needle = normalise(label)
            if not needle or needle in haystack:
                continue
            # Every token must appear as a WHOLE WORD, short ones included.
            # Substring matching, or dropping tokens under three characters, both
            # made "Item c" count as covered because "item" appeared somewhere --
            # a false "covered" is the one error this check exists to prevent.
            tokens = [t for t in needle.split() if t not in filler]
            if tokens and all(
                re.search(rf"\b{re.escape(token)}\b", haystack) for token in tokens
            ):
                continue
            missing.append(label)
        return missing


def plan_enumeration(user_id: str, question: str) -> EnumerationPlan | None:
    """Read the manifest for the authoritative list this question asks for.

    Returns None when the question is not an enumeration, or when nothing in the
    manifest credibly backs one. **No list is better than a wrong list**: the
    coverage check enforces whatever this returns, so a plausible-but-wrong set
    would be actively harmful rather than merely unhelpful.
    """
    if not user_id or not is_enumeration(question):
        return None

    head = head_noun(question)
    if not head:
        return None

    best: tuple[float, EnumerationPlan] | None = None
    for term in enumeration_terms(question):
        # Every candidate term must contain the head noun. Without this rule,
        # "list every approval required" matched the group "Required Trainings" on
        # the word "required" and confidently returned seven trainings as the list
        # of approvals -- which the coverage check would then have enforced.
        if head not in term.split():
            continue

        rows = [
            row
            for row in db.find_manifest_items(user_id, term.split()[-1])
            if word_match(term, row["item"])
            or word_match(term, row["group_label"])
            or word_match(term, row["group_context"])
        ]
        if len(rows) < config.MANIFEST_MIN_ITEMS:
            continue

        groups: dict[tuple, list[dict]] = {}
        for row in rows:
            groups.setdefault((row["document_id"], row["group_label"], row["kind"]), []).append(row)

        for (document_id, group_label, kind), matched in groups.items():
            # Whether to expand to the WHOLE group depends on what the term matched.
            #
            # If the term names the GROUP -- "fault code" matching "Diagnostic Fault
            # Codes" -- every member is in scope, and expanding is what makes "there
            # are 9" authoritative rather than "9 of them contained the word".
            #
            # If the term matched individual ITEMS -- "leave" matching "Sick Leave"
            # and "Parental Leave" inside the group "Employee Handbook" -- expanding
            # is wrong: it would answer "all types of leave" with every section of
            # the handbook, Compensation Bands included. Measured exactly that: 6
            # items for a document with 3 types of leave.
            names_group = word_match(term, group_label) or word_match(
                term, matched[0].get("group_context", "")
            )
            items = (
                db.manifest_group_items(user_id, document_id, group_label, kind) or matched
                if names_group
                else matched
            )
            if len(items) < config.MANIFEST_MIN_ITEMS:
                continue

            # Relevance, not size. A group is preferred because the query names it,
            # not because it is long -- size alone let an unrelated 16-item group
            # win. The phrase bonus is what lets "record type" (which names the
            # classification table by its column header, 7 rows) beat bare "record"
            # (which matched 2 row labels inside it).
            score = (
                (2.0 if names_group else 0.0)
                + len(matched) / max(len(items), 1)
                + 0.25 * len(term.split())
            )
            total = len(items)
            plan = EnumerationPlan(
                term=term,
                items=items[: config.MANIFEST_MAX_ITEMS],
                truncated_from=total if total > config.MANIFEST_MAX_ITEMS else 0,
            )
            if best is None or score > best[0]:
                best = (score, plan)

    return best[1] if best else None


# ---------------------------------------------------------------------------
# Use 2: heading-path filtering
# ---------------------------------------------------------------------------


def heading_filter(user_id: str, question: str) -> list[str]:
    """Heading paths a question explicitly names, for pre-search narrowing.

    "what does the eligibility section say?" should not search the whole corpus. A
    named section is an exact structural constraint, and applying it as a Chroma
    ``where`` clause shrinks the candidate pool *before* any similarity is computed
    -- which is both faster and more precise than hoping the ranker prefers the
    right section.

    Returns [] unless the question names a section that really exists. A filter
    built from a guess would silently exclude the answer, which is far worse than
    not filtering: the failure would look like the corpus not containing it.
    """
    if not user_id or not config.HEADING_FILTER_ENABLED:
        return []

    match = _SECTION_REFERENCE.search(question or "")
    if not match:
        return []
    name = (match.group("name") or match.group("named") or "").strip()
    # Strip leading question words the regex may have swallowed ("does the scope").
    tokens = [t for t in name.split() if t.lower() not in _STOPWORDS]
    if not tokens:
        return []
    name = " ".join(tokens)
    if len(name) < 3:
        return []

    paths = db.all_heading_paths(user_id)
    if not paths:
        return []
    matched = [
        row["heading_path"]
        for row in paths
        if word_match(name, row["heading_path"].split(">")[-1])
    ]
    if not matched:
        return []

    # A filter that keeps most of the corpus has narrowed nothing and can only
    # exclude something relevant, so it is dropped.
    if len(matched) > max(1, int(len(paths) * config.HEADING_FILTER_MAX_SHARE)):
        return []
    return sorted(set(matched))
