"""V3.2: tables as structured objects, stitched across pages, chunked by row.

Why overlap cannot fix tables
-----------------------------
Chunk overlap is a *character* remedy. It repeats the tail of one chunk at the
head of the next, which recovers a severed sentence and does nothing at all for a
table, because what a table fragment is missing is not the previous 120 characters
-- it is the **header row**, which may be hundreds of rows and an entire page
away. Two failures follow, and both are in this corpus:

* ``widgetx_operations_manual.pdf`` -- fault codes E-01..E-03 sit in a table at
  the bottom of page 7 and E-04..E-09 continue at the top of page 8. The page-8
  fragment has no header row at all, so nothing in it says that "E-04 | Firmware
  checksum mismatch | Reflash firmware" is *code, meaning, action*. Measured: the
  known-answer question "what does fault code E-04 mean" fails under BOTH V2 and
  V3.1.
* ``onboarding_guide.pdf`` -- the training-deadline table breaks the same way
  between pages 4 and 5.

So the fix has to be structural, and it has to happen at ingest: reassembling a
table at query time would mean a second retrieval round trip on every hit, and
would still be guessing at which fragments belong together.

The path
--------
1. **Extract** every table on every page with PyMuPDF ``find_tables()`` as rows +
   column headers + page, keeping each fragment's bounding box and whether it was
   the first/last content on its page.
2. **Stitch** consecutive-page fragments into one logical table (see
   ``_continues`` for the rule, which needs no pixel thresholds).
3. **Chunk by row**, never mid-row, repeating the header block in every part.
4. **Summarise** each table as a one-line description carrying the same
   ``table_id``, so dense retrieval can find a table by meaning rather than by
   matching a query against pipe-delimited digits.

Retrieval then expands any hit into the whole table by metadata fetch
(``retrieval.expand_table_siblings``).

Table regions are already excluded from the prose stream by
``ingestion._extract_page_items`` (it skips text blocks covered by a table's
bbox), so nothing is double-processed. With this flag ON the rendered table block
is additionally kept out of the text chunk stream, because the table path owns it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from langchain.docstore.document import Document

import backend.config as config
from backend.chunk_schema import apply_defaults
from backend.ingestion import _is_heading, clean_text, count_tokens

# Cell separator, matching the V2 renderer so a table's text shape does not change
# just because the flag moved.
CELL = " | "

# Metadata key that marks a summary chunk. Kept as a bool because Chroma stores
# scalars and because retrieval filters on it. Unused by the grouped-JSON path
# (there is no separate summary chunk any more -- see TABLE_CHUNK_FORMAT below),
# kept for the legacy pipe-format path it was built for.
SUMMARY_FLAG = "is_table_summary"

# V4: delimiter for entities_in_group, deliberately different from
# chunk_schema.LIST_DELIMITER -- this field is a citation aid read by code, not
# a Layer B list, and does not need the extra readability of a space.
ENTITY_DELIM = ","

# Metadata marker written ONLY by the grouped-JSON path. Its presence (or
# absence) is how backend.tables.assemble tells a table's two possible chunk
# shapes apart, and how the migration script tells a migrated table_id from a
# stale one.
TABLE_CHUNK_FORMAT = "table_chunk_format"
GROUPED_JSON_FORMAT = "grouped_json_v1"

# V4 bugfix: header names that mark a column as an alternate row label (an
# ID's paired name), narrow and explicit on purpose -- pairing every second
# column with the first would glue a fault code to its "Meaning" description.
_ALIAS_NAME_HEADERS = frozenset(
    {"name", "full name", "employee name", "customer name", "vendor name", "contact name"}
)


@dataclass
class TableFragment:
    """One ``find_tables()`` hit on one page, before stitching."""

    rows: list[list[str]]
    page: int
    y0: float
    y1: float
    first_on_page: bool
    last_on_page: bool
    section: str = ""

    @property
    def columns(self) -> int:
        return max((len(r) for r in self.rows), default=0)


@dataclass
class LogicalTable:
    """One table after continuations have been merged."""

    source: str
    headers: list[str]
    rows: list[list[str]] = field(default_factory=list)
    row_pages: list[int] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    section: str = ""
    index: int = 0
    fragments: int = 1

    @property
    def table_id(self) -> str:
        """Stable id: document + ordinal.

        Not a content hash -- a re-ingested document whose table gained a row must
        keep the same id, or the old parts survive under an orphaned id and a
        sibling fetch reassembles a mix of both versions.
        """
        return f"{self.source}#t{self.index}"

    @property
    def page_range(self) -> str:
        if self.page_end > self.page_start:
            return f"{self.page_start}-{self.page_end}"
        return str(self.page_start)

    @property
    def spans_pages(self) -> bool:
        return self.page_end > self.page_start


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value: str | None) -> str:
    return (value or "").replace("\n", " ").strip()


def render_row(row: list[str]) -> str:
    return CELL.join(_cell(c) for c in row)


def render_header(headers: list[str]) -> str:
    """The header block repeated at the top of every part.

    Two lines: the names, then a dashed rule. The rule is what makes it
    unambiguous to the answer model that these are column labels and not the
    table's first data row -- without it, a part that begins "Code | Meaning |
    Required action" reads as another row of codes.
    """
    names = render_row(headers)
    if not names:
        return ""
    return f"{names}\n{CELL.join('---' for _ in headers)}"


def _is_blank_row(row: list[str]) -> bool:
    return not any(_cell(c) for c in row)


def _same_row(left: list[str], right: list[str]) -> bool:
    """Case- and whitespace-insensitive row equality, for repeated headers."""
    norm = lambda row: [re.sub(r"\s+", " ", _cell(c)).casefold() for c in row]
    return norm(left) == norm(right)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _chrome_lines(doc: fitz.Document) -> set[str]:
    """Running headers/footers, by the same frequency rule the text path uses.

    Needed because "was this table the last thing on its page?" must ignore the
    footer that appears on every page. Reimplemented over raw blocks rather than
    reusing ``ingestion._strip_running_lines`` because that one consumes cleaned,
    table-filtered page text and this pass runs before any of that exists.
    """
    per_page: list[set[str]] = []
    for index in range(doc.page_count):
        lines: list[str] = []
        for block in doc.load_page(index).get_text("blocks"):
            for line in (block[4] or "").split("\n"):
                line = line.strip()
                if line and len(line) <= 120:
                    lines.append(line)
        # Only page margins are candidates, as in the text path: a line repeated
        # mid-page is far more likely to be real content.
        per_page.append(set(lines[:2] + lines[-3:]))

    if len(per_page) < 3:
        return set()
    counts: dict[str, int] = {}
    for lines in per_page:
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(2, int(doc.page_count * config.HEADER_FOOTER_MIN_PAGE_RATIO))
    chrome = {line for line, n in counts.items() if n >= threshold}
    # A bare page number is chrome on every document, whatever its frequency.
    return chrome


def _is_chrome(text: str, chrome: set[str]) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    return any(line.strip() in chrome for line in stripped.split("\n") if line.strip())


def _section_for(fragment: TableFragment, headings: list[str]) -> str:
    """The nearest heading above ``fragment`` that is not the table's own header row.

    A borderless table's header row is sometimes emitted as a prose block that
    reaches outside the detected table bbox, so it survives the bbox filter and
    then matches ``_is_heading`` -- short, capitalised, no terminal punctuation.
    The relocation policy's cover-page version table was captioned
    "Version Number Creation Date Modified By Approved by", i.e. its own columns.

    Walking back past such a line yields the real enclosing heading, or "" when
    there genuinely is none (a cover-page table above the first heading).
    """
    if not fragment.rows:
        return headings[-1] if headings else ""
    signature = re.sub(r"\s+", " ", " ".join(_cell(c) for c in fragment.rows[0])).casefold()
    for heading in reversed(headings):
        if re.sub(r"\s+", " ", heading).casefold() != signature:
            return heading
    return ""


def _page_spans(page: fitz.Page) -> list[tuple[float, float, float, float, str]]:
    """Every non-blank text span on ``page``, one entry per PyMuPDF span.

    A span, not a word: PyMuPDF's ``dict``-mode text already groups a
    multi-word cell ("Employee 1", "Manager 5", "Quarterly performance review
    record #1") into a single span per visual line, so reading spans avoids
    re-splitting a multi-word value the way word-by-word text does.

    The clip is deliberately wider than the page's own mediabox: PyMuPDF's
    ``get_text()`` silently drops any glyph positioned outside it even with no
    ``clip`` argument at all, which is exactly what a page laid out wider than
    its own margins (as on ``Large_Multi_Page_Tables_Test.pdf``, where the
    leftmost column sits at a negative x-coordinate and two-digit values in
    the rightmost column run past the right edge) needs recovered -- confirmed
    directly: the unclipped default returns "MP0001" and "...record #1" for a
    cell whose actual content, recoverable only with a wider clip, is
    "EMP0001" and "...record #10".
    """
    wide = fitz.Rect(-150, 0, page.rect.width + 150, page.rect.height)
    spans = []
    for block in page.get_text("dict", clip=wide)["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text and text.strip():
                    x0, y0, x1, y1 = span["bbox"]
                    spans.append((x0, y0, x1, y1, text))
    return spans


def _orphan_spans(
    page: fitz.Page, rect: fitz.Rect
) -> list[tuple[float, float, float, float, str]]:
    """Spans inside ``rect``'s row band but outside its column band.

    ``find_tables()``'s default line-based strategy sizes a table's bbox off
    detected ruling/gridlines. A borderless table has none, so the bbox comes
    from text alignment instead -- and on
    ``Large_Multi_Page_Tables_Test.pdf`` it silently excludes the leftmost and
    rightmost columns (the leftmost header even sits at a negative x-coordinate,
    off the printable margin). A span that falls within the table's row band
    but outside its detected column span is proof a column was dropped.
    """
    return [
        (x0, y0, x1, y1, text)
        for x0, y0, x1, y1, text in _page_spans(page)
        if y0 >= rect.y0 - 1
        and y1 <= rect.y1 + 1
        and (x1 < rect.x0 - 2 or x0 > rect.x1 + 2)
    ]


def _recover_borderless_columns(
    table,
    rect: fitz.Rect,
    rows: list[list[str]],
    orphans: list[tuple[float, float, float, float, str]],
) -> list[list[str]] | None:
    """Reattach the column(s) ``orphans`` proves ``find_tables()`` dropped.

    Deliberately does NOT re-detect the table with PyMuPDF's ``strategy="text"``
    -- tried first, it fabricated characters that do not exist in the PDF's own
    content stream (turned the real "MP0001" into "EMP0001" near this
    document's negative-x-coordinate margin). Instead this reuses
    ``table.rows[i].bbox`` -- PyMuPDF's own per-row y-band, already proven
    correct for the columns it did detect -- to slot each orphan span into the
    right row, by row order rather than by re-running detection at all.
    """
    if len(rows) != len(table.rows):
        return None  # extract() and .rows should always be 1:1; don't guess if not

    left = sorted((o for o in orphans if o[2] <= rect.x0 + 2), key=lambda o: o[0])
    right = sorted((o for o in orphans if o[0] >= rect.x1 - 2), key=lambda o: o[0])
    if not left and not right:
        return None

    def cell_for_row(pool, y0: float, y1: float) -> str:
        parts = [text for ox0, oy0, ox1, oy1, text in pool if oy0 >= y0 - 1 and oy1 <= y1 + 1]
        return " ".join(parts).strip()

    new_rows = []
    for row, table_row in zip(rows, table.rows):
        _, y0, _, y1 = table_row.bbox
        new_row = list(row)
        if left:
            new_row = [cell_for_row(left, y0, y1)] + new_row
        if right:
            new_row = new_row + [cell_for_row(right, y0, y1)]
        new_rows.append(new_row)

    # Only trust the recovery if every row that had content to begin with --
    # including the header -- actually got a value for the recovered
    # column(s); a row-band mismatch that leaves a real row's cell blank is
    # worse than the dropped column it was meant to fix. A row that was
    # already entirely blank stays blank either way and is not evidence of a
    # bad recovery -- it is filtered out downstream regardless.
    header = new_rows[0]
    if left and not header[0].strip():
        return None
    if right and not header[-1].strip():
        return None
    for original, new_row in zip(rows[1:], new_rows[1:]):
        if _is_blank_row(original):
            continue
        if (left and not new_row[0].strip()) or (right and not new_row[-1].strip()):
            return None
    return new_rows


def extract_fragments(pdf_doc: fitz.Document, source: str) -> list[TableFragment]:
    """Extract every table on every page, in reading order, with page context.

    ``first_on_page`` / ``last_on_page`` ignore running headers and footers, which
    is what lets the stitching rule below work without any pixel thresholds:
    on page 7 of the operations manual the fault-code table is followed only by
    "Acme Corporation - Internal Use Only" and a page number, and on page 8 the
    continuation is preceded only by the running title.
    """
    chrome = _chrome_lines(pdf_doc)
    fragments: list[TableFragment] = []
    section = ""
    headings: list[str] = []

    for index in range(pdf_doc.page_count):
        page = pdf_doc.load_page(index)
        page_no = index + 1

        try:
            found = list(page.find_tables().tables)
        except Exception:
            # Table detection is best-effort; a failure must not lose the page.
            found = []

        rects = [fitz.Rect(t.bbox) for t in found]
        # Non-chrome prose blocks, used both for section tracking and for the
        # first/last-on-page tests.
        prose: list[tuple[float, float, str]] = []
        for block in page.get_text("blocks"):
            rect = fitz.Rect(block[0], block[1], block[2], block[3])
            text = block[4] or ""
            if _is_chrome(text, chrome):
                continue
            area = abs(rect.get_area())
            if area > 0 and any(
                abs((rect & r).get_area()) / area > 0.5 for r in rects
            ):
                continue  # inside a table region
            prose.append((rect.y0, rect.y1, text))
        prose.sort(key=lambda item: item[0])

        page_fragments: list[TableFragment] = []
        for table, rect in zip(found, rects):
            try:
                rows = [list(r) for r in table.extract()]
            except Exception:
                continue
            orphans = _orphan_spans(page, rect)
            if orphans:
                # Before the blank-row filter: recovery relies on rows and
                # table.rows staying 1:1 so a row-band bbox lines up with the
                # right extracted row.
                recovered = _recover_borderless_columns(table, rect, rows, orphans)
                if recovered is not None:
                    rows = recovered
            rows = [r for r in rows if not _is_blank_row(r)]
            if not rows:
                continue
            page_fragments.append(
                TableFragment(
                    rows=rows,
                    page=page_no,
                    y0=rect.y0,
                    y1=rect.y1,
                    first_on_page=not any(y1 <= rect.y0 for y0, y1, _ in prose),
                    last_on_page=not any(y0 >= rect.y1 for y0, y1, _ in prose),
                )
            )

        # Section attribution: walk the page's prose and tables together in
        # reading order so each table inherits the heading above it.
        stream: list[tuple[float, str, TableFragment | str]] = [
            (y0, "prose", text) for y0, _, text in prose
        ] + [(f.y0, "table", f) for f in page_fragments]
        stream.sort(key=lambda item: item[0])
        for _, kind, payload in stream:
            if kind == "prose":
                cleaned = clean_text(str(payload))
                for line in cleaned.split("\n"):
                    if _is_heading(line):
                        section = line
                        headings.append(line)
            else:
                assert isinstance(payload, TableFragment)
                payload.section = _section_for(payload, headings)
                fragments.append(payload)

    for fragment in fragments:
        fragment.rows = [r for r in fragment.rows if not _is_blank_row(r)]
    return [f for f in fragments if f.rows]


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def _continues(previous: TableFragment, candidate: TableFragment) -> bool:
    """Is ``candidate`` a continuation of ``previous`` onto the next page?

    Four structural conditions, no pixel thresholds:

    1. Consecutive pages.
    2. Same column count -- a different shape is a different table.
    3. ``previous`` ran to the bottom of its page (nothing but chrome below it).
    4. ``candidate`` starts at the top of its page (nothing but chrome above it).

    Together these cover BOTH continuation forms the brief names, and the two
    forms need no separate test: if the candidate's first row repeats the previous
    table's header it is dropped as a duplicate (``merge`` does that), and if it
    does not, every row is data. Asking "does this row look like a header?" is not
    needed and would be unreliable -- PyMuPDF reports the fault-code
    continuation's ``header.names`` as ``['E-04', ...]``, i.e. it simply takes row
    zero, so its own header detection cannot distinguish the two cases either.

    The residual risk is two genuinely unrelated tables of equal width, one ending
    a page and the next opening the following page with no prose between them at
    all. Conditions 3 and 4 make that rare: an unrelated table is nearly always
    introduced by a heading or a lead-in sentence, which is prose above it.
    """
    return (
        candidate.page == previous.page + 1
        and candidate.columns == previous.columns
        and previous.last_on_page
        and candidate.first_on_page
    )


def stitch(fragments: list[TableFragment], source: str) -> list[LogicalTable]:
    """Merge page-break continuations into logical tables, in document order."""
    tables: list[LogicalTable] = []
    previous: TableFragment | None = None

    for fragment in fragments:
        if previous is not None and _continues(previous, fragment):
            table = tables[-1]
            rows = fragment.rows
            # Repeated-header form: the continuation restates the column names.
            if table.headers and rows and _same_row(rows[0], table.headers):
                rows = rows[1:]
            table.rows.extend(rows)
            table.row_pages.extend([fragment.page] * len(rows))
            table.page_end = fragment.page
            table.fragments += 1
        else:
            headers = fragment.rows[0] if fragment.rows else []
            body = fragment.rows[1:]
            tables.append(
                LogicalTable(
                    source=source,
                    headers=[_cell(c) for c in headers],
                    rows=body,
                    row_pages=[fragment.page] * len(body),
                    page_start=fragment.page,
                    page_end=fragment.page,
                    section=fragment.section,
                    index=len(tables),
                )
            )
        previous = fragment

    return [t for t in tables if t.rows or t.headers]


def extract_tables(pdf_path, source: str) -> list[LogicalTable]:
    """Extract and stitch every table in a PDF. Returns [] on any read failure."""
    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception:
        return []
    try:
        if pdf_doc.is_encrypted and not pdf_doc.authenticate(""):
            return []
        return stitch(extract_fragments(pdf_doc, source), source)
    except Exception:
        # Tables are an enhancement; losing them must never lose the document.
        return []
    finally:
        pdf_doc.close()


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summary_line(table: LogicalTable, max_labels: int | None = None) -> str:
    """One-line natural-language description of what a table is about.

    Derived, not generated by a model: the enclosing section heading says what the
    table is, the column names say what it records, and the first-column values say
    which things it covers. That is the whole of what makes a table findable, it
    costs no tokens at ingest, and it is reproducible -- an LLM-written caption
    would differ between two ingests of the same document and make the
    known-answer comparison unrepeatable.

    Row LABELS are included; cell VALUES never are. A label is what a question
    names ("what does E-04 mean", "the anti-bribery deadline") so it drives
    retrieval, while a value is what an answer asserts. Keeping values out means
    this generated line cannot be the source of a figure even if it reaches the
    prompt -- which is the rule that answers and citations come from real chunk
    text.
    """
    limit = max_labels if max_labels is not None else config.TABLE_SUMMARY_MAX_LABELS

    parts = [f"Table: {table.section}" if table.section else "Table"]
    columns = [c for c in table.headers if c]
    if columns:
        parts.append("columns: " + ", ".join(columns))

    labels = [_cell(row[0]) for row in table.rows if row and _cell(row[0])]
    labels = list(dict.fromkeys(labels))
    if labels:
        shown = labels[:limit]
        suffix = ", ..." if len(labels) > len(shown) else ""
        parts.append("rows: " + ", ".join(shown) + suffix)

    parts.append(f"{len(table.rows)} rows; page {table.page_range}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Row-wise chunking
# ---------------------------------------------------------------------------


@dataclass
class TablePart:
    """One chunk of a table: the header block plus a whole number of rows."""

    text: str
    page_start: int
    page_end: int
    rows: int


def chunk_table(table: LogicalTable, max_tokens: int | None = None) -> list[TablePart]:
    """Split a table on ROW boundaries, repeating the header block in every part.

    A row is never split: half a row loses the cell that gives the other half its
    meaning. A single row that exceeds the budget on its own is therefore emitted
    oversized rather than cut -- the same trade the V2 path already makes for a
    whole table.
    """
    budget = max_tokens or config.TABLE_CHUNK_MAX_TOKENS or config.CHUNK_SIZE
    header = render_header(table.headers)

    def render_part(rows: list[str]) -> str:
        body = "\n".join(rows)
        return f"{header}\n{body}" if header else body

    parts: list[TablePart] = []
    buffer: list[str] = []
    pages: list[int] = []

    def flush() -> None:
        nonlocal buffer, pages
        if not buffer:
            return
        parts.append(
            TablePart(
                text=render_part(buffer),
                page_start=min(pages),
                page_end=max(pages),
                rows=len(buffer),
            )
        )
        buffer = []
        pages = []

    for row, page in zip(table.rows, table.row_pages):
        rendered = render_row(row)
        if not rendered:
            continue
        # Measured on the JOINED part, not as a running sum of per-row counts.
        # Tokenisation is not additive -- the newlines and the header block cost
        # tokens the per-row sum never sees -- and summing overshot the budget by
        # enough that a part measured 60 tokens actually held 68. Tables are tens
        # of rows, so re-tokenising the candidate is affordable at ingest and
        # exact is worth more here than fast.
        if buffer and count_tokens(render_part(buffer + [rendered])) > budget:
            flush()
        buffer.append(rendered)
        pages.append(page)
    flush()

    if not parts and header:
        # A header-only table (every row blank) still deserves to be findable.
        parts.append(
            TablePart(
                text=header,
                page_start=table.page_start,
                page_end=table.page_end,
                rows=0,
            )
        )
    return parts


# ---------------------------------------------------------------------------
# V4: grouped-JSON row chunking
# ---------------------------------------------------------------------------


@dataclass
class TableGroup:
    """One chunk of a table under the grouped-JSON scheme: a whole number of
    rows, each rendered as a ``{column: value}`` object."""

    rows: list[dict]
    row_start: int
    row_end: int
    page_start: int
    page_end: int
    entities: list[str]


def _row_object(headers: list[str], row: list[str]) -> dict:
    """One row as a JSON object keyed by column header.

    A blank or missing header falls back to ``colN`` (1-indexed) rather than
    dropping the cell -- a ragged row must not silently lose a value just
    because its column has no name.
    """
    obj: dict[str, str] = {}
    for index, cell in enumerate(row):
        name = _cell(headers[index]) if index < len(headers) else ""
        obj[name or f"col{index + 1}"] = _cell(cell)
    return obj


def group_table_rows_json(
    table: LogicalTable,
    rows_per_chunk: int | None = None,
    max_tokens: int | None = None,
    description: str = "",
) -> list[TableGroup]:
    """Group a table's rows into JSON-array chunks.

    ``rows_per_chunk`` (default ``config.TABLE_ROWS_PER_CHUNK``) is a target,
    not a hard cap: a group flushes early, before reaching it, if adding the
    next row would push the rendered chunk (description included) over the
    token budget -- the same trade ``chunk_table`` makes for pipe-rendered
    parts, just measured on JSON instead. A single row that alone exceeds the
    budget is still emitted rather than split; a row is the smallest unit that
    keeps its cells meaningful.
    """
    limit_rows = rows_per_chunk or config.TABLE_ROWS_PER_CHUNK
    budget = max_tokens or config.TABLE_CHUNK_MAX_TOKENS or config.CHUNK_SIZE

    # V4 bugfix: a bare first-column value ("Employee ID": "EMP0034") only
    # names a row one way, and a question can just as easily name it "Employee
    # 34" (the second column). Paired only when column 1's header is
    # explicitly name-shaped -- NOT for every second column, or a fault-code
    # table's "Meaning" description would get wrongly glued to its code.
    second_header = table.headers[1].strip().casefold() if len(table.headers) > 1 else ""
    pair_second_column = second_header in _ALIAS_NAME_HEADERS

    def render(rows: list[dict]) -> str:
        body = json.dumps(rows, ensure_ascii=False)
        return f"{description}\n{body}" if description else body

    groups: list[TableGroup] = []
    buffer: list[dict] = []
    pages: list[int] = []
    indices: list[int] = []

    def flush() -> None:
        nonlocal buffer, pages, indices
        if not buffer:
            return
        entities: list[str] = []
        for obj in buffer:
            values = list(obj.values())
            label = values[0] if values else ""
            if pair_second_column and len(values) > 1 and values[1] and values[1] != label:
                label = f"{label}:{values[1]}" if label else values[1]
            if label and label not in entities:
                entities.append(label)
        groups.append(
            TableGroup(
                rows=list(buffer),
                row_start=indices[0],
                row_end=indices[-1],
                page_start=min(pages),
                page_end=max(pages),
                entities=entities,
            )
        )
        buffer, pages, indices = [], [], []

    for position, (row, page) in enumerate(zip(table.rows, table.row_pages)):
        obj = _row_object(table.headers, row)
        if not any(v for v in obj.values()):
            continue
        candidate = buffer + [obj]
        if buffer and (
            len(candidate) > limit_rows or count_tokens(render(candidate)) > budget
        ):
            flush()
        buffer.append(obj)
        pages.append(page)
        indices.append(position)
    flush()

    if not groups:
        # A header-only table (every row blank) still deserves to be findable.
        groups.append(
            TableGroup(
                rows=[],
                row_start=0,
                row_end=-1,
                page_start=table.page_start,
                page_end=table.page_end,
                entities=[],
            )
        )
    return groups


def build_table_documents_grouped_json(
    tables: list[LogicalTable],
    doc_title: str,
    content_hash: str,
    heading_paths: dict[int, str] | None = None,
    rows_per_chunk: int | None = None,
    max_tokens: int | None = None,
) -> list[Document]:
    """V4: one Document per row-group, JSON-encoded, no separate summary chunk.

    Each group's ``page_content`` is the table's one-line description (the same
    text ``summary_line`` produces for the legacy path, folded in here instead
    of shipped as its own chunk) followed by the group's rows as a JSON array.
    Column names live inside every row object, so nothing needs repeating the
    way the pipe-format header block did.

    Metadata keeps the sibling-linking fields ``table_id``/``table_part``/
    ``table_total_parts``/``page_range``/``table_headers`` at GROUP granularity
    (one group = one part), and adds ``entities_in_group`` -- the group's
    first-column values, for entity-name retrieval without a per-row chunk.
    ``table_chunk_format`` marks these as the new shape.
    """
    out: list[Document] = []
    for table in tables:
        heading_path = (heading_paths or {}).get(table.index, "")
        base = _base_metadata(table, doc_title, heading_path)
        description = summary_line(table)
        groups = group_table_rows_json(table, rows_per_chunk, max_tokens, description)
        total = len(groups)

        for number, group in enumerate(groups, start=1):
            body = json.dumps(group.rows, ensure_ascii=False)
            out.append(
                Document(
                    page_content=f"{description}\n{body}",
                    metadata={
                        **base,
                        "page": group.page_start,
                        "page_end": group.page_end,
                        "table_part": number,
                        "table_total_parts": total,
                        "entities_in_group": ENTITY_DELIM.join(group.entities),
                        TABLE_CHUNK_FORMAT: GROUPED_JSON_FORMAT,
                        SUMMARY_FLAG: False,
                        "content_hash": content_hash,
                    },
                )
            )
    return out


# ---------------------------------------------------------------------------
# Documents for the index
# ---------------------------------------------------------------------------


def _base_metadata(table: LogicalTable, doc_title: str, heading_path: str) -> dict:
    """Metadata shared by every chunk of one table. Scalars only (Chroma)."""
    return apply_defaults({
        "source": table.source,
        "doc_title": doc_title,
        "section": table.section,
        "section_title": table.section,
        "heading_path": heading_path,
        "has_table": True,
        "table_id": table.table_id,
        "page_range": table.page_range,
        # Delimited string, not a list: Chroma metadata values must be scalar.
        "table_headers": CELL.join(c for c in table.headers if c),
        "table_rows": len(table.rows),
        "table_spans_pages": table.spans_pages,
        "table_fragments": table.fragments,
        # V3.3: a table part is a table -- typed structurally, not by asking a
        # model to confirm what the extractor already proved.
        "content_type": "table",
    })


def build_table_documents(
    tables: list[LogicalTable],
    doc_title: str,
    content_hash: str,
    heading_paths: dict[int, str] | None = None,
    max_tokens: int | None = None,
) -> list[Document]:
    """Build the part chunks and the summary chunk for every table.

    Each part carries the sibling metadata the brief requires -- ``table_id``
    (shared), ``table_part``, ``table_total_parts``, ``page_range``,
    ``table_headers`` -- and the summary chunk carries the same ``table_id`` so a
    hit on it expands to the real rows.

    ``chunk_index`` is left unset here; ``ingestion.build_chunks`` assigns it once
    table and text chunks have been merged into document order.
    """
    out: list[Document] = []
    for table in tables:
        heading_path = (heading_paths or {}).get(table.index, "")
        base = _base_metadata(table, doc_title, heading_path)
        parts = chunk_table(table, max_tokens)
        total = len(parts)

        # The summary goes FIRST, before the parts. Two reasons: it reads as the
        # table's introduction, and it keeps the group's own page numbers
        # non-decreasing -- a summary carries the table's START page, so emitting
        # it last put a page-7 chunk after a page-8 one and made chunk_index
        # disagree with reading order, which is what backend.context sorts by.
        out.append(
            Document(
                page_content=summary_line(table),
                metadata={
                    **base,
                    "page": table.page_start,
                    "page_end": table.page_end,
                    "table_part": 0,
                    "table_total_parts": total,
                    SUMMARY_FLAG: True,
                    "content_hash": content_hash,
                },
            )
        )

        for number, part in enumerate(parts, start=1):
            label = (
                f"{base['section']} (table, page {table.page_range}, "
                f"part {number} of {total})"
                if total > 1
                else f"{base['section']} (table, page {table.page_range})"
            )
            out.append(
                Document(
                    # The label is provenance, not data: it names the table and
                    # says which part this is, and holds no cell value. A part
                    # that reads as a bare grid of pipes is one the answer model
                    # cannot place in the document.
                    page_content=f"{label.strip()}\n{part.text}",
                    metadata={
                        **base,
                        "page": part.page_start,
                        "page_end": part.page_end,
                        "table_part": number,
                        "table_total_parts": total,
                        SUMMARY_FLAG: False,
                        "content_hash": content_hash,
                    },
                )
            )
    return out


# ---------------------------------------------------------------------------
# Reassembly at retrieval time
# ---------------------------------------------------------------------------

_RULE = re.compile(r"^(?:-{2,}\s*\|\s*)*-{2,}$")
_PART_LABEL = re.compile(r"\(table, page\b")


def header_block_from_metadata(metadata: dict) -> str:
    """Rebuild the header block from a part's ``table_headers`` metadata."""
    names = str(metadata.get("table_headers") or "").strip()
    if not names:
        return ""
    columns = [c for c in names.split(CELL.strip()) if c.strip()] or [names]
    return f"{names}\n{CELL.join('---' for _ in columns)}"


def part_rows(text: str, header_names: str) -> list[str]:
    """Strip a part's label and repeated header block, leaving its data rows.

    The header is repeated in every part on purpose (a fragment without it is
    useless), which means reassembly has to remove the repeats or the model sees
    the column names interleaved through the rows.
    """
    rows: list[str] = []
    for position, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if not stripped:
            continue
        if position == 0 and _PART_LABEL.search(stripped):
            continue
        if header_names and stripped == header_names.strip():
            continue
        if _RULE.match(stripped):
            continue
        rows.append(stripped)
    return rows


@dataclass
class AssembledTable:
    """A table rebuilt from its sibling chunks, ready for the prompt."""

    text: str
    page_start: int
    page_end: int
    parts_included: int
    parts_total: int
    complete: bool


def _assemble_legacy(
    records: list[dict],
    matched_parts: set[int],
    limit_parts: int,
    limit_tokens: int,
) -> AssembledTable | None:
    """Legacy pipe-format reassembly: rebuild from repeated-header row parts.

    ``records`` are ``{"document", "metadata"}`` dicts straight from the metadata
    fetch, summary chunk included. The summary is used only when the caps bite --
    it is generated text, so it never stands in for rows that could have been
    included.

    Beyond ``max_parts`` or ``max_tokens`` the result is the header block, the
    one-line summary, and the parts that actually matched. That is a deliberate
    degradation: a 40-part table would otherwise evict every other document from
    the context window, and the parts that matched are the ones the question is
    about.
    """
    parts = sorted(
        (r for r in records if not r["metadata"].get(SUMMARY_FLAG)),
        key=lambda r: int(r["metadata"].get("table_part") or 0),
    )
    if not parts:
        return None

    summaries = [r for r in records if r["metadata"].get(SUMMARY_FLAG)]
    first = parts[0]["metadata"]
    header_names = str(first.get("table_headers") or "")
    header = header_block_from_metadata(first)
    total = int(first.get("table_total_parts") or len(parts))
    section = str(first.get("section") or "")
    page_range = str(first.get("page_range") or "")

    def label(complete: bool, shown: list[int] | None = None) -> str:
        what = f"Table: {section}" if section else "Table"
        where = f" (page {page_range})" if page_range else ""
        if complete:
            return what + where
        parts_shown = ", ".join(str(n) for n in (shown or []))
        return f"{what}{where} -- excerpt, part{'s' if len(shown or []) != 1 else ''} " \
               f"{parts_shown} of {total}"

    def render(chosen: list[dict], complete: bool, note: str = "") -> AssembledTable:
        rows: list[str] = []
        for record in chosen:
            rows.extend(part_rows(record["document"] or "", header_names))
        pages = [int(r["metadata"].get("page") or 0) for r in chosen] or [0]
        ends = [
            int(r["metadata"].get("page_end") or r["metadata"].get("page") or 0)
            for r in chosen
        ]
        shown = [int(r["metadata"].get("table_part") or 0) for r in chosen]
        heading = label(complete, shown)
        body = "\n".join(line for line in [heading, note, header, *rows] if line)
        return AssembledTable(
            text=body,
            page_start=min(pages),
            page_end=max(ends or pages),
            parts_included=len(chosen),
            parts_total=total,
            complete=complete,
        )

    full = render(parts, complete=True)
    if len(parts) <= limit_parts and count_tokens(full.text) <= limit_tokens:
        return full

    kept = [
        r for r in parts if int(r["metadata"].get("table_part") or 0) in matched_parts
    ] or parts[:1]
    # The summary says what the WHOLE table covers, which is what the excerpt
    # cannot. Its "Table: " prefix is dropped so it does not read as a second
    # heading under the one above it.
    note = ""
    if summaries:
        note = re.sub(r"^Table:\s*", "", summaries[0]["document"] or "").strip()
        note = f"Full table summary -- {note}" if note else ""
    return render(kept[:limit_parts], complete=False, note=note)


def _parse_group_rows(document: str) -> list[dict]:
    """Recover a grouped-JSON chunk's row objects from its ``page_content``.

    The content is ``"<description>\\n<json array>"`` -- the description is
    exactly one line (``summary_line`` never emits an embedded newline), so
    splitting on the first ``\\n`` isolates the JSON blob unambiguously.
    """
    _, _, blob = (document or "").partition("\n")
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _assemble_grouped_json(
    records: list[dict],
    matched_parts: set[int],
    limit_parts: int,
    limit_tokens: int,
) -> AssembledTable | None:
    """V4 reassembly: concatenate row-objects from sibling groups, in order.

    No header block to de-duplicate -- every row object already carries its own
    column names -- so this is simpler than the legacy path: parse each group's
    JSON array, concatenate in ``table_part`` order, and re-render as one array.
    """
    groups = sorted(records, key=lambda r: int(r["metadata"].get("table_part") or 0))
    if not groups:
        return None

    first = groups[0]["metadata"]
    total = int(first.get("table_total_parts") or len(groups))
    section = str(first.get("section") or "")
    page_range = str(first.get("page_range") or "")
    description = (groups[0]["document"] or "").split("\n", 1)[0]

    def label(complete: bool, shown: list[int]) -> str:
        what = f"Table: {section}" if section else "Table"
        where = f" (page {page_range})" if page_range else ""
        if complete:
            return what + where
        parts_shown = ", ".join(str(n) for n in shown)
        return f"{what}{where} -- excerpt, part{'s' if len(shown) != 1 else ''} " \
               f"{parts_shown} of {total}"

    def render(chosen: list[dict], complete: bool) -> AssembledTable:
        rows: list[dict] = []
        for record in chosen:
            rows.extend(_parse_group_rows(record["document"] or ""))
        pages = [int(r["metadata"].get("page") or 0) for r in chosen] or [0]
        ends = [
            int(r["metadata"].get("page_end") or r["metadata"].get("page") or 0)
            for r in chosen
        ]
        shown = [int(r["metadata"].get("table_part") or 0) for r in chosen]
        heading = label(complete, shown)
        body = json.dumps(rows, ensure_ascii=False)
        text = "\n".join(line for line in [heading, description, body] if line)
        return AssembledTable(
            text=text,
            page_start=min(pages),
            page_end=max(ends or pages),
            parts_included=len(chosen),
            parts_total=total,
            complete=complete,
        )

    full = render(groups, complete=True)
    if len(groups) <= limit_parts and count_tokens(full.text) <= limit_tokens:
        return full

    kept = [
        r for r in groups if int(r["metadata"].get("table_part") or 0) in matched_parts
    ] or groups[:1]
    return render(kept[:limit_parts], complete=False)


def assemble(
    records: list[dict],
    matched_parts: set[int],
    max_parts: int | None = None,
    max_tokens: int | None = None,
) -> AssembledTable | None:
    """Rebuild one logical table from the sibling chunks fetched by ``table_id``.

    Dispatches on ``table_chunk_format``: a table's chunks are either ALL
    grouped-JSON (written by ``build_table_documents_grouped_json``) or ALL
    legacy pipe-format (written by ``build_table_documents``) -- a single table
    is never split across both, because the migration script and
    ``build_chunks`` always replace a table's chunks as one unit. Records are
    routed to whichever reassembly the majority carry, so a caller need not know
    which path ingested this table.
    """
    limit_parts = max_parts if max_parts is not None else config.TABLE_SIBLING_MAX_PARTS
    limit_tokens = (
        max_tokens if max_tokens is not None else config.TABLE_SIBLING_MAX_TOKENS
    )
    grouped = [r for r in records if r["metadata"].get(TABLE_CHUNK_FORMAT) == GROUPED_JSON_FORMAT]
    if grouped:
        return _assemble_grouped_json(grouped, matched_parts, limit_parts, limit_tokens)
    return _assemble_legacy(records, matched_parts, limit_parts, limit_tokens)


def tables_hash(tables: list[LogicalTable]) -> str:
    """Digest of every table's rendered content, folded into the document hash.

    Turning the flag on changes the chunk stream, so it MUST invalidate the
    "unchanged, skipped" check -- otherwise a corpus stays indexed under the old
    path while the run reports success.
    """
    digest = hashlib.sha256()
    for table in tables:
        digest.update(table.table_id.encode("utf-8"))
        digest.update(render_header(table.headers).encode("utf-8"))
        for row in table.rows:
            digest.update(render_row(row).encode("utf-8"))
    return digest.hexdigest()
