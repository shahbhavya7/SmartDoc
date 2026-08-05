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
# scalars and because retrieval filters on it.
SUMMARY_FLAG = "is_table_summary"


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


def assemble(
    records: list[dict],
    matched_parts: set[int],
    max_parts: int | None = None,
    max_tokens: int | None = None,
) -> AssembledTable | None:
    """Rebuild one logical table from the sibling chunks fetched by ``table_id``.

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
    limit_parts = max_parts if max_parts is not None else config.TABLE_SIBLING_MAX_PARTS
    limit_tokens = (
        max_tokens if max_tokens is not None else config.TABLE_SIBLING_MAX_TOKENS
    )

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
