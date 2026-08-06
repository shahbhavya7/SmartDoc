"""PDF ingestion: structure-aware extraction, cleaning, and chunking.

This module turns PDFs into two granularities of ``Document`` chunk:

* **child chunks** (~``CHILD_CHUNK_SIZE`` tokens) -- what gets embedded and
  searched. Small chunks embed cleanly because they cover one topic.
* **parent chunks** (~``PARENT_CHUNK_SIZE`` tokens) -- what the model actually
  reads. A child is precise enough to *find* but usually too small to *answer*
  from, so retrieval returns children and context assembly substitutes parents.

Why this replaced fixed per-page 800-token chunking
---------------------------------------------------
The previous implementation split **each page independently**. Any section
spanning a page break was severed at the boundary, and overlap never crossed
pages, so multi-hop and synthesis questions could not see a continuous section.
Measured: the old pipeline produced **0** chunks spanning a page break --
structurally impossible. This one produces 124 on the same corpus.

Structure preserved as metadata
-------------------------------
* ``section`` -- the nearest enclosing heading, so a chunk starting mid-section
  still states which policy it belongs to. The breadcrumb
  ``"<doc title> > <section>"`` is also prepended to the embedded text, which
  improves retrieval because the topic words are present even in a chunk that
  never repeats them.
* ``page`` / ``page_end`` -- the true page range covered.
* ``parent_id`` -- for parent expansion at answer time.
* ``prev_id`` / ``next_id`` -- for neighbour expansion.
* ``has_table`` -- tables are never split mid-row, and are marked so
  extraction-style prompts can be told a table is present.
* ``content_hash`` -- lets ingestion skip unchanged documents.

Tables are extracted with PyMuPDF's table finder and rendered as pipe-delimited
rows. The prose extractor then skips blocks covered by a table's bounding box,
so table cells are not also emitted as scrambled prose -- which is how the
previous ``get_text("text")`` path destroyed every table in the corpus.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import backend.config as config
from backend.chunk_schema import apply_defaults

PDF_SUFFIX = ".pdf"

# Separator hierarchy: paragraph -> line -> sentence -> clause -> word. The
# empty-string fallback only triggers for a single token longer than the whole
# chunk budget, which does not occur in prose.
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

_ENCODING = "cl100k_base"

# A heading is a short line that opens with a section number, or a short
# Title/UPPER case line with no terminal punctuation. Deliberately
# conservative: a false positive fragments a section, which is worse than a
# false negative (the chunk simply inherits the previous heading).
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.{0,80})$")
_MAX_HEADING_CHARS = 90

# A ticket/asset-register row rendered as a flat text line ("Hardware Request
# P3 03-Jan-2026" wrapped and truncated to "Hardware Request P3 03-") passes
# the title-case heuristic below -- every word capitalised, no terminal
# punctuation -- exactly like a real heading. Measured on a real IT support
# manual ingested on the plain-text path: five such rows got promoted into
# the manifest's heading tree alongside the document's genuine "2.3.1
# Hardware Request" section, inflating an 8-item enumeration to 15. A
# priority code (P1-P4) followed by a trailing digit-hyphen fragment is not
# a pattern any real heading in this corpus produces.
_TICKET_ROW_RE = re.compile(r"\bP[1-4]\b.*\d+-\s*$")


class PDFReadError(Exception):
    """Raised when a PDF cannot be located, opened, or read."""


@dataclass(frozen=True)
class PageText:
    """Extracted, cleaned text for a single PDF page."""

    page: int
    text: str


@dataclass
class Block:
    """One structural unit of a document, in reading order.

    ``heading_path`` is populated only by the V3.1 markdown path, where the
    document's real heading hierarchy is known ("Leave Policy > Sick Leave >
    Eligibility"). The plain-text path leaves it empty, which is what keeps every
    downstream behaviour identical when MARKDOWN_INGESTION_ENABLED is off.
    """

    kind: str  # "heading" | "paragraph" | "table"
    text: str
    page: int
    section: str = ""
    heading_path: str = ""


@dataclass
class ParsedDocument:
    """A fully parsed PDF: title, page texts, and the block stream."""

    source: str
    title: str
    page_count: int
    pages: list[PageText] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    content_hash: str = ""

    # "text" for the V2 plain-text path, "markdown" when markdown conversion
    # succeeded. Stored on the document row as well, so a PDF that degraded to
    # the fallback path stays identifiable long after ingest.
    extraction_mode: str = "text"

    # Path to the cached markdown, relative to PROJECT_ROOT; "" on the text path.
    markdown_path: str = ""

    # V3.2: tables lifted out of the text stream as structured objects, already
    # stitched across page breaks. Empty unless TABLE_AWARE_INGESTION_ENABLED --
    # with the flag off, tables stay inline in ``blocks`` exactly as in V2.
    # Typed loosely to keep ``backend.tables`` from importing this module and back.
    tables: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder = None


def _enc():
    """Return a cached tiktoken encoder (construction is not free)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Return the token count of ``text`` under the embedding tokenizer."""
    return len(_enc().encode(text))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_pdf_path(pdf_path: str | Path) -> Path:
    """Validate that ``pdf_path`` points to an existing ``.pdf`` file."""
    path = Path(pdf_path)
    if not path.exists():
        raise PDFReadError(f"File not found: {path}")
    if not path.is_file():
        raise PDFReadError(f"Not a file: {path}")
    if path.suffix.lower() != PDF_SUFFIX:
        raise PDFReadError(f"Not a PDF (expected .pdf suffix): {path}")
    return path


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_text(raw_text: str) -> str:
    """Conservatively clean text extracted from a PDF fragment.

    Repairs only what PDF extraction reliably breaks: line-break hyphenation,
    mid-paragraph line wraps, runs of horizontal whitespace, and lines that are
    only a page number.

    Content-based header/footer detection needs document-wide frequency
    information and is handled by ``_strip_running_lines``.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Line-break hyphenation: "classifica-\ntion" -> "classification".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # The same artifact after the extractor has already turned the line break
    # into a space: "authent- ication" -> "authentication". PyMuPDF does this
    # inside a text block, so the newline form alone leaves the word broken --
    # which made a real corpus fact unfindable by exact search. Restricted to
    # lowercase-to-lowercase so it cannot damage bullet markers ("- Airfare"),
    # numeric ranges ("20 - 30"), or hyphenated proper nouns.
    text = re.sub(r"([a-z])-[ \t]+([a-z])", r"\1\2", text)

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        lines.append(stripped)

    paragraphs: list[str] = []
    current: list[str] = []
    bullet_re = re.compile(r"^([-•*]|\d+[.)]|[A-Za-z][.)])\s+")

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if bullet_re.match(line):
            flush()
            paragraphs.append(line)
            continue
        current.append(line)
    flush()

    joined = "\n".join(p for p in paragraphs if p)
    return re.sub(r"[ \t]{2,}", " ", joined).strip()


def _strip_running_lines(page_lines: list[list[str]], min_ratio: float) -> set[str]:
    """Identify running headers/footers by cross-page repetition.

    Frequency is a far safer signal than pattern-matching for "Page X of Y" or a
    company name: body sentences do not recur verbatim across most pages.
    """
    if len(page_lines) < 3:
        return set()

    counts: dict[str, int] = {}
    for lines in page_lines:
        # Only page margins are candidates; a repeated line mid-page is far
        # more likely to be real content.
        candidates = lines[:2] + lines[-3:]
        for line in {c for c in candidates if c and len(c) <= 120}:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, int(len(page_lines) * min_ratio))
    return {line for line, n in counts.items() if n >= threshold}


# ---------------------------------------------------------------------------
# Structure-aware extraction
# ---------------------------------------------------------------------------


def _is_heading(text: str) -> bool:
    """Return True if ``text`` looks like a section heading."""
    if not text or len(text) > _MAX_HEADING_CHARS or "\n" in text:
        return False
    if _NUMBERED_HEADING.match(text):
        # "1. Scope" is a heading; "Step 1: Collect itemised receipts for every
        # expense." is a procedure step -- length and terminal punctuation are
        # what separate them.
        return not text.endswith(".") or len(text.split()) <= 6
    if text.endswith((".", ":", ";", ",")):
        return False
    if _TICKET_ROW_RE.search(text):
        return False
    words = text.split()
    if not 1 <= len(words) <= 12:
        return False
    if text.isupper():
        return True
    capitalised = sum(1 for w in words if w[:1].isupper())
    return capitalised >= max(2, len(words) - 1)


def _table_to_text(rows: list[list[str | None]]) -> str:
    """Render extracted table rows as pipe-delimited lines.

    Kept as one block so a table is never split mid-row: splitting turns
    "E-07 | Door interlock open" into two unrelated chunks and makes exhaustive
    extraction impossible.
    """
    lines: list[str] = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if not any(cells):
            continue
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _covered_by_table(rect: fitz.Rect, table_rects: list[fitz.Rect]) -> bool:
    """Return True if ``rect`` lies substantially inside any table region."""
    area = abs(rect.get_area())
    if area <= 0:
        return False
    for table_rect in table_rects:
        if abs((rect & table_rect).get_area()) / area > 0.5:
            return True
    return False


def _extract_page_items(
    page: fitz.Page, emit_tables: bool = True
) -> list[tuple[str, str]]:
    """Extract one page as ordered ``(kind, text)`` items.

    Tables are extracted first and their bounding boxes recorded; prose blocks
    overlapping a table region are then skipped, so table cells are not emitted
    twice. Blocks are ordered top-to-bottom, approximating reading order for
    single- and multi-column layouts alike.

    ``emit_tables=False`` (V3.2, TABLE_AWARE_INGESTION_ENABLED) still detects
    tables and still excludes their regions from the prose -- what changes is that
    the rendered table is NOT put into the text chunk stream, because
    ``backend.tables`` owns it from that point on. Detection cannot be skipped:
    without the bounding boxes, table cells reappear as scrambled prose.
    """
    items: list[tuple[float, str, str]] = []

    table_rects: list[fitz.Rect] = []
    try:
        for table in page.find_tables().tables:
            rect = fitz.Rect(table.bbox)
            table_rects.append(rect)
            if not emit_tables:
                continue
            rendered = _table_to_text(table.extract())
            if rendered:
                items.append((rect.y0, "table", rendered))
    except Exception:
        # Table detection is best-effort; a failure must not lose the page.
        table_rects = []

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not text or not text.strip():
            continue
        if _covered_by_table(fitz.Rect(x0, y0, x1, y1), table_rects):
            continue
        cleaned = clean_text(text)
        if cleaned:
            items.append((y0, "paragraph", cleaned))

    items.sort(key=lambda item: item[0])
    return [(kind, text) for _, kind, text in items]


def extract_document(pdf_path: str | Path) -> ParsedDocument:
    """Parse a PDF into a page-numbered stream of structural blocks.

    V3.2: when ``TABLE_AWARE_INGESTION_ENABLED`` is on, tables are lifted out of
    the block stream into ``ParsedDocument.tables`` as structured, page-stitched
    objects. Their regions are still excluded from the prose, so nothing is
    double-processed. With the flag off this function is unchanged.

    Raises:
        PDFReadError: for a missing file, a non-PDF, an encrypted PDF that will
            not open with a blank password, or an unreadable file.
    """
    path = _require_pdf_path(pdf_path)

    try:
        doc = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise PDFReadError(f"Could not open PDF {path.name}: {exc}") from exc

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PDFReadError(f"PDF {path.name} is encrypted and could not be opened.")

        table_aware = config.TABLE_AWARE_INGESTION_ENABLED
        raw_pages: list[list[tuple[str, str]]] = []
        for index in range(doc.page_count):
            raw_pages.append(
                _extract_page_items(doc.load_page(index), emit_tables=not table_aware)
            )

        page_lines = [
            [
                line
                for kind, text in items
                if kind == "paragraph"
                for line in text.split("\n")
            ]
            for items in raw_pages
        ]
        running = _strip_running_lines(page_lines, config.HEADER_FOOTER_MIN_PAGE_RATIO)

        blocks: list[Block] = []
        pages: list[PageText] = []
        section = ""
        title = ""
        title_taken = False

        for page_no, items in enumerate(raw_pages, start=1):
            page_texts: list[str] = []
            for kind, text in items:
                if kind == "table":
                    blocks.append(
                        Block(kind="table", text=text, page=page_no, section=section)
                    )
                    page_texts.append(text)
                    continue

                # Page 1 is exempt from running-line stripping. A document's
                # title block is usually the same string as its running header,
                # so stripping by frequency would delete the title itself --
                # losing the most useful piece of context for the breadcrumb.
                kept = [
                    line
                    for line in text.split("\n")
                    if line and (page_no == 1 or line not in running)
                ]
                for para in kept:
                    # A line that repeats across pages is chrome, not structure.
                    # Only the FIRST such line on page 1 may become a heading --
                    # that is the title block. Every other repeating line (a
                    # footer like "Internal Use Only") must never become a
                    # section, or chunks inherit the footer as their heading.
                    is_running = para in running
                    may_head = _is_heading(para) and (not is_running or not title_taken)

                    if may_head:
                        if is_running:
                            title_taken = True
                        section = para
                        if not title:
                            title = para
                        blocks.append(
                            Block(
                                kind="heading", text=para, page=page_no, section=section
                            )
                        )
                    else:
                        blocks.append(
                            Block(
                                kind="paragraph",
                                text=para,
                                page=page_no,
                                section=section,
                            )
                        )
                    page_texts.append(para)

            page_text = "\n".join(page_texts).strip()
            if page_text:
                pages.append(PageText(page=page_no, text=page_text))

        source = path.name
        # Addendum 2 needs the same structured tables to fill the relational cell
        # store, so extraction runs for EITHER flag -- one parse serves both. What
        # each flag controls is what happens downstream: table-aware ingestion
        # changes the chunk stream (``build_chunks`` gates on its own flag), while
        # the SQL store only reads ``parsed.tables`` and leaves chunking alone.
        if table_aware or config.PARALLEL_SQL_LOOKUP_ENABLED:
            # Imported here, not at module scope: backend.tables depends on this
            # module's cleaner and heading test, so a top-level import would be
            # circular.
            from backend.tables import extract_tables, tables_hash

            structured_tables = extract_tables(path, source)
        else:
            structured_tables = []

        # The digest covers the tables as well as the prose. Turning the flag on
        # moves table text from the block stream into its own chunks, so the hash
        # MUST change -- otherwise "unchanged, skipped" leaves the corpus indexed
        # under the old path while the run reports success.
        #
        # Salted only when the document actually HAS tables. A table-free document
        # produces a byte-identical chunk stream either way, so salting it
        # regardless would re-embed nine of this corpus's thirteen documents to
        # arrive at exactly what was already indexed.
        full_text = "\n".join(b.text for b in blocks)
        if structured_tables:
            full_text = f"tables\n{tables_hash(structured_tables)}\n{full_text}"

        return ParsedDocument(
            source=source,
            title=title or path.stem.replace("_", " ").title(),
            page_count=doc.page_count,
            pages=pages,
            blocks=blocks,
            content_hash=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
            tables=structured_tables,
        )
    except PDFReadError:
        raise
    except Exception as exc:
        raise PDFReadError(f"Failed reading pages of {path.name}: {exc}") from exc
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Build a token-length splitter over the configured separators."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )


def _fallback_heading(title: str, section: str) -> str:
    """A heading path composed from the V1 flat section, for the text path.

    V3.3 makes ``heading_path`` a contract field every chunk carries, and a filter
    is only useful if it is populated. The plain-text path has no hierarchy, but it
    does know the document title and the nearest heading -- the same information
    the V1 extractor already had -- so composing them gives a real, if shallow,
    path instead of an empty string.

    This deliberately supersedes one V3.1 guarantee: flag-OFF chunk METADATA is no
    longer byte-identical to V2. Chunk text, boundaries and counts are unchanged;
    only this field gains a value it can be filtered and cited by.
    """
    parts = [p for p in ((title or "").strip(), (section or "").strip()) if p]
    # The first chunk's section is often the title itself, and
    # "Handbook > Handbook" is a worse label than "Handbook".
    if len(parts) == 2 and parts[0].casefold() == parts[1].casefold():
        parts = parts[:1]
    return " > ".join(parts)


def _breadcrumb(title: str, section: str, heading_path: str = "") -> str:
    """Build the topical prefix prepended to a chunk before embedding.

    With a real heading hierarchy available (V3.1 markdown path) the whole path is
    used, not just the leaf: a chunk under "Leave Policy > Sick Leave >
    Eligibility" should carry "sick leave" even though its own heading says only
    "Eligibility". The document title is dropped when the path already opens with
    it, which markdown H1s usually do -- repeating it wastes tokens and dilutes
    the embedding.
    """
    if heading_path:
        parts = [p.strip() for p in heading_path.split(">") if p.strip()]
        if parts and title and parts[0].casefold() == title.casefold():
            parts = parts[1:]
        return " > ".join([p for p in (title,) if p] + parts)
    return " > ".join(p for p in (title, section) if p)


@dataclass
class Chunk:
    """An intermediate chunk with its structural provenance.

    ``blocks`` is retained so a parent can be re-chunked into children at a
    smaller budget without flattening to text first -- which would lose the
    per-block page numbers and make every child inherit the parent's start page.
    """

    text: str
    page_start: int
    page_end: int
    section: str
    has_table: bool
    blocks: list[Block] = field(default_factory=list)
    heading_path: str = ""


def _group_key(block: Block) -> str:
    """The identity a section run is grouped by.

    The full heading path, when one is known, rather than the leaf title: two
    different sections can both be titled "Eligibility", and grouping by leaf
    title alone would silently merge them into one parent chunk. On the plain-text
    path ``heading_path`` is empty and this is exactly the old ``block.section``.
    """
    return block.heading_path or block.section


def _group_sections(blocks: list[Block]) -> list[list[Block]]:
    """Group the block stream into runs sharing an enclosing section."""
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_key: str | None = None
    for block in blocks:
        key = _group_key(block)
        if current_key is None or key == current_key:
            current.append(block)
            current_key = key
            continue
        groups.append(current)
        current = [block]
        current_key = key
    if current:
        groups.append(current)
    return groups


def _chunk_blocks(
    blocks: list[Block], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split a run of blocks into chunks that never split a table.

    A table block is emitted whole even if it exceeds the budget: half a table
    loses its header row and becomes uninterpretable, which is worse than an
    oversized chunk.
    """
    chunks: list[Chunk] = []
    buffer: list[Block] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = "\n".join(b.text for b in buffer).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    page_start=min(b.page for b in buffer),
                    page_end=max(b.page for b in buffer),
                    section=buffer[0].section,
                    has_table=any(b.kind == "table" for b in buffer),
                    blocks=list(buffer),
                    heading_path=buffer[0].heading_path,
                )
            )
        buffer = []
        buffer_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block.text)

        if block.kind == "table":
            flush()
            chunks.append(
                Chunk(
                    text=block.text,
                    page_start=block.page,
                    page_end=block.page,
                    section=block.section,
                    has_table=True,
                    blocks=[block],
                    heading_path=block.heading_path,
                )
            )
            continue

        if block_tokens > chunk_size:
            flush()
            for piece in _splitter(chunk_size, chunk_overlap).split_text(block.text):
                piece = piece.strip()
                if piece:
                    chunks.append(
                        Chunk(
                            text=piece,
                            page_start=block.page,
                            page_end=block.page,
                            section=block.section,
                            has_table=False,
                            blocks=[block],
                            heading_path=block.heading_path,
                        )
                    )
            continue

        if buffer_tokens + block_tokens > chunk_size:
            flush()
        buffer.append(block)
        buffer_tokens += block_tokens

    flush()
    return chunks


def _splice_by_page(
    children: list[Document], table_documents: list[Document]
) -> list[Document]:
    """Insert table chunks into the text stream at their page position.

    Insertion, not a re-sort of everything: the text children are already in
    reading order, and re-sorting them by page would reorder a chunk that legally
    spans a page break (its ``page`` is the range's START, so it can sort ahead of
    a chunk that begins later on the previous page).

    Each table's chunks stay contiguous and in part order, placed after every text
    chunk that starts on the table's first page or earlier. A page number cannot
    order two chunks *within* a page, so "after everything on page 7" is the
    finest placement available -- and it is the right one for the corpus's real
    cases, where a table that breaks across a page sits at the bottom of the first
    one. Inserting before the page's prose instead made chunk_index disagree with
    page order, which is what ``backend.context`` sorts a document's units by.
    """
    if not table_documents:
        return children

    groups: dict[str, list[Document]] = {}
    for document in table_documents:
        groups.setdefault(document.metadata["table_id"], []).append(document)

    def page_of(document: Document) -> int:
        return int(document.metadata.get("page") or 0)

    pending = sorted(groups.values(), key=lambda group: page_of(group[0]))
    out: list[Document] = []
    for child in children:
        while pending and page_of(pending[0][0]) < page_of(child):
            out.extend(pending.pop(0))
        out.append(child)
    for group in pending:
        out.extend(group)
    return out


def _table_heading_paths(parsed: ParsedDocument) -> dict[int, str]:
    """Map each table's ordinal to a V3.1 heading path, when one is known.

    Tables are found by PyMuPDF and know only their nearest heading, so the full
    hierarchy is recovered by matching that heading against the markdown sections
    already parsed for this document. Empty on the plain-text path, which has no
    hierarchy to offer -- the same V2-neutral value every other V3.1 key takes.
    """
    if parsed.extraction_mode != "markdown":
        return {}
    by_title: dict[str, str] = {}
    for block in parsed.blocks:
        if block.section and block.heading_path:
            by_title.setdefault(block.section, block.heading_path)
    return {
        table.index: by_title.get(table.section, "")
        for table in parsed.tables
        if table.section
    }


def build_chunks(
    parsed: ParsedDocument,
    child_size: int | None = None,
    child_overlap: int | None = None,
    parent_size: int | None = None,
) -> tuple[list[Document], list[Document]]:
    """Build parent and child chunks for a parsed document.

    Child text is prefixed with a ``"<title> > <section>"`` breadcrumb before
    embedding. This costs a handful of tokens and improves retrieval: a chunk
    deep inside "3. Annual Leave Entitlement" that never repeats the words
    "annual leave" is otherwise near-invisible to a query that uses them.

    Returns:
        ``(parents, children)`` as LangChain ``Document`` lists.
    """
    child_size = child_size or config.CHILD_CHUNK_SIZE
    child_overlap = (
        child_overlap if child_overlap is not None else config.CHILD_CHUNK_OVERLAP
    )
    parent_size = parent_size or config.PARENT_CHUNK_SIZE

    parents: list[Document] = []
    children: list[Document] = []
    parent_index = 0
    child_index = 0

    for group in _group_sections(parsed.blocks):
        for parent_chunk in _chunk_blocks(group, parent_size, 0):
            parent_id = f"{parsed.source}#p{parent_index}"
            parents.append(
                Document(
                    page_content=parent_chunk.text,
                    metadata={
                        "id": parent_id,
                        "source": parsed.source,
                        "doc_title": parsed.title,
                        "section": parent_chunk.section,
                        "page": parent_chunk.page_start,
                        "page_end": parent_chunk.page_end,
                        "parent_index": parent_index,
                        "has_table": parent_chunk.has_table,
                        "heading_path": parent_chunk.heading_path,
                        "section_title": parent_chunk.section,
                    },
                )
            )

            if parent_chunk.has_table:
                # Never re-split a table; the parent is also the child.
                child_chunks = [parent_chunk]
            else:
                # Re-chunk the parent's own blocks (not its flattened text) so
                # each child keeps the true page range of the blocks it covers,
                # and a child may legitimately span a page break.
                child_chunks = _chunk_blocks(
                    parent_chunk.blocks, child_size, child_overlap
                )

            for child in child_chunks:
                breadcrumb = _breadcrumb(
                    parsed.title, child.section, child.heading_path
                )
                text = f"{breadcrumb}\n\n{child.text}" if breadcrumb else child.text
                children.append(
                    Document(
                        page_content=text,
                        # V3.3: apply_defaults guarantees the FULL contract on
                        # every chunk -- the table_* and Layer B fields included,
                        # at their defaults for a prose chunk. A field may be
                        # empty; it may never be absent, or a `where` clause on it
                        # silently under-returns instead of erroring.
                        metadata=apply_defaults({
                            "source": parsed.source,
                            "doc_title": parsed.title,
                            "section": child.section,
                            "page": child.page_start,
                            "page_end": child.page_end,
                            "chunk_index": child_index,
                            "parent_id": parent_id,
                            "has_table": child.has_table,
                            "token_count": count_tokens(text),
                            "content_hash": parsed.content_hash,
                            # V3.1. Both are "" on the plain-text path, so a
                            # flag-OFF chunk's metadata is unchanged in value.
                            # Delimited string, not a list: Chroma stores
                            # scalars only.
                            "heading_path": child.heading_path
                            or _fallback_heading(parsed.title, child.section),
                            "section_title": child.section,
                            "extraction_mode": parsed.extraction_mode,
                        }),
                    )
                )
                child_index += 1
            parent_index += 1

    # V3.2: table chunks are their own stream, spliced back into reading order by
    # page. They deliberately have no ``parent_id``: a table part is already a
    # complete unit (its header block is repeated in every part), and substituting
    # a prose parent for it would replace the rows with the surrounding text.
    # Completeness comes from sibling expansion instead.
    # Gated on the flag, not merely on ``parsed.tables`` being populated: with
    # Addendum 2 on and table-aware chunking off, tables are extracted for the
    # SQL store alone and must NOT change the chunk stream. Keying off the flag
    # is what keeps flag-off chunk output byte-identical.
    if parsed.tables and config.TABLE_AWARE_INGESTION_ENABLED:
        # V4: grouped-JSON row chunks replace the legacy pipe-format ones once
        # GROUPED_JSON_TABLE_CHUNKS_ENABLED is on. The legacy builder stays
        # available (and is what the V3.2 tests/verify script still exercise)
        # for as long as the flag can still be measured against itself.
        if config.GROUPED_JSON_TABLE_CHUNKS_ENABLED:
            from backend.tables import build_table_documents_grouped_json as _build_tables
        else:
            from backend.tables import build_table_documents as _build_tables

        children = _splice_by_page(
            children,
            _build_tables(
                parsed.tables,
                doc_title=parsed.title,
                content_hash=parsed.content_hash,
                heading_paths=_table_heading_paths(parsed),
            ),
        )
        # chunk_index is reassigned over the merged stream: it is what
        # backend.context sorts a document's units by to restore reading order,
        # so a table appearing at index 400 in a document that ends at page 8
        # would put the table after every paragraph in the assembled prompt.
        for position, child in enumerate(children):
            child.metadata["chunk_index"] = position

    # Neighbour links, assigned after all children exist so ordering is
    # document-global rather than per-parent.
    for position, child in enumerate(children):
        child.metadata["prev_id"] = (
            f"{parsed.source}:{position - 1}" if position > 0 else ""
        )
        child.metadata["next_id"] = (
            f"{parsed.source}:{position + 1}" if position < len(children) - 1 else ""
        )

    return parents, children


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_document(
    pdf_path: str | Path,
) -> tuple[ParsedDocument, list[Document], list[Document]]:
    """Parse and chunk one PDF end to end. Returns ``(parsed, parents, children)``."""
    parsed = extract_document(pdf_path)
    parents, children = build_chunks(parsed)
    return parsed, parents, children


def _pdfs_in(data_dir: str | Path) -> list[Path]:
    """Return sorted PDFs directly inside ``data_dir`` (no recursion)."""
    directory = Path(data_dir)
    if not directory.exists():
        raise PDFReadError(f"Data directory not found: {directory}")
    if not directory.is_dir():
        raise PDFReadError(f"Not a directory: {directory}")
    pdf_paths = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == PDF_SUFFIX
    )
    if not pdf_paths:
        raise PDFReadError(f"No PDF files found in {directory}")
    return pdf_paths


def load_directory(data_dir: str | Path) -> tuple[list[Document], list[Document]]:
    """Parse and chunk every PDF directly inside ``data_dir``.

    Subdirectories are not traversed, which is how ``data/legacy_v1/`` stays out
    of the index.
    """
    all_parents: list[Document] = []
    all_children: list[Document] = []
    for pdf_path in _pdfs_in(data_dir):
        _, parents, children = load_document(pdf_path)
        all_parents.extend(parents)
        all_children.extend(children)
    return all_parents, all_children


# ---------------------------------------------------------------------------
# Backward-compatible single-granularity API
# ---------------------------------------------------------------------------


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract cleaned, page-numbered text (non-empty pages only)."""
    return extract_document(pdf_path).pages


def chunk_pages(
    pages: list[PageText],
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split page texts into flat chunks with ``{source, page, chunk_index}``."""
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    splitter = _splitter(size, overlap)

    chunks: list[Document] = []
    running_index = 0
    for page in pages:
        for piece in splitter.split_text(page.text):
            content = piece.strip()
            if not content:
                continue
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "page": page.page,
                        "chunk_index": running_index,
                    },
                )
            )
            running_index += 1
    return chunks


def load_and_chunk(
    pdf_path: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Flat single-granularity chunking of one PDF."""
    path = _require_pdf_path(pdf_path)
    return chunk_pages(extract_pages(path), path.name, chunk_size, chunk_overlap)


def load_and_chunk_directory(
    data_dir: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Flat single-granularity chunking of every PDF in a directory."""
    out: list[Document] = []
    for pdf_path in _pdfs_in(data_dir):
        out.extend(load_and_chunk(pdf_path, chunk_size, chunk_overlap))
    return out
