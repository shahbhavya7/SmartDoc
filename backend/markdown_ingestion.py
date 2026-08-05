"""V3.1: markdown conversion at ingest, and heading-aware two-stage splitting.

Why markdown before chunking
----------------------------
The V2 path reconstructs structure by *guessing*: a short line without terminal
punctuation is treated as a heading (``ingestion._is_heading``). That heuristic is
deliberately conservative, so it finds the obvious headings and misses nested
ones, and it can never recover a hierarchy -- every heading is flat, so a chunk
knows its nearest heading but not which policy that heading sits under.

``pymupdf4llm`` reads the PDF's own font/size structure and emits ``#``/``##``/
``###``. That turns "which section is this?" from an inference into a fact, and it
renders tables as pipe-delimited markdown rows rather than as prose fragments.

The two-stage splitter
----------------------
1. ``MarkdownHeaderTextSplitter`` cuts on real heading boundaries and hands back
   the heading hierarchy for each section. A boundary here is a *semantic* one --
   a section ends where the document says it ends.
2. Only sections still over the configured budget (the locked ~800/120) go
   through the existing ``RecursiveCharacterTextSplitter``. A section under budget
   is kept whole, which is the point: it never gets cut mid-sentence at an
   arbitrary character count.

The output is a ``ParsedDocument`` in exactly the V2 shape -- a page-numbered
``Block`` stream -- so ``build_chunks`` and everything downstream of it (parent/
child granularity, table protection, neighbour links, citations) run unchanged.
The only difference is that each block now also carries ``heading_path``.

Page numbers
------------
Markdown is generated per page (``page_chunks=True``) and every line is kept
paired with the page it came from. Sections are matched back to those lines with a
forward cursor, so a section spanning a page break gets a true ``page``/
``page_end`` range and citations keep pointing at the right page. This is why the
converter is not simply called once over the whole document: a single markdown
blob has no page information at all, and every citation would be unusable.

Fallback
--------
Conversion failure, a near-empty result, or markdown that recovers materially
less text than the plain-text extractor all fall back to the V2 path, and the
document is marked ``extraction_mode = "text"``. Scanned / image-only PDFs land
here -- they have no text layer for either path, and the honest outcome is the
existing "NO TEXT EXTRACTED" report rather than a silent success on a page of
stray glyphs.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

import backend.config as config
from backend.ingestion import (
    Block,
    PageText,
    ParsedDocument,
    PDFReadError,
    _require_pdf_path,
    _splitter,
    _strip_running_lines,
    count_tokens,
    extract_document,
)

logger = logging.getLogger("smartdoc.markdown_ingestion")

# A markdown table row, header separator (``|---|---|``) included.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
# Leading ATX markers and any closing run, e.g. "## Sick Leave ##" -> "Sick Leave".
_HEADING_MARKERS = re.compile(r"^\s*#{1,6}\s*|\s*#+\s*$")
# A list item or a numbered step: its own paragraph, never merged with the line
# above it. Mirrors ingestion.clean_text's bullet handling so an enumerated list
# does not collapse into one run-on paragraph.
_BULLET = re.compile(r"^\s*(?:[-•*+]|\d+[.)]|[A-Za-z][.)])\s+")

# Unscoped ingest runs (the evaluation harness) still need somewhere to cache
# markdown, and it must not be a directory a real user's id could collide with.
UNSCOPED_DIR_NAME = "_unscoped"


class MarkdownConversionError(Exception):
    """Raised when a PDF cannot be converted to usable markdown."""


@dataclass
class Section:
    """One heading-delimited section of a document, with its page-tagged lines."""

    heading_path: str
    title: str
    lines: list[tuple[str, int]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line for line, _ in self.lines).strip()


# ---------------------------------------------------------------------------
# Stage 0: PDF -> markdown
# ---------------------------------------------------------------------------


def _clean_markdown_line(line: str) -> str:
    """Repair extraction artifacts without touching markdown structure.

    ``ingestion.clean_text`` cannot be reused here: it reflows wrapped lines into
    paragraphs and drops digit-only lines, which would flatten a markdown table
    into one long line and delete a table cell that holds only a number. Only the
    hyphenation repairs carry over -- they are the artifact that made a real
    corpus fact unfindable by exact search (see DECISIONS.md), and they are
    line-local.
    """
    line = line.replace("\r", "")
    # "authent- ication" -> "authentication". Lowercase-to-lowercase only, so
    # bullet markers, numeric ranges, and hyphenated proper nouns survive.
    return re.sub(r"([a-z])-[ \t]+([a-z])", r"\1\2", line.rstrip())


def convert_to_markdown(pdf_path: str | Path) -> tuple[list[tuple[str, int]], int]:
    """Convert a PDF to markdown lines, each tagged with its source page.

    Returns:
        ``(lines, page_count)`` where ``lines`` is ``[(text, page_number), ...]``
        in reading order.

    Raises:
        MarkdownConversionError: if the converter is unavailable or fails.
    """
    path = _require_pdf_path(pdf_path)

    try:
        import pymupdf4llm
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise MarkdownConversionError(
            "pymupdf4llm is not installed; markdown ingestion cannot run."
        ) from exc

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise MarkdownConversionError(f"Could not open PDF {path.name}: {exc}") from exc

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise MarkdownConversionError(f"PDF {path.name} is encrypted.")
        page_count = doc.page_count
        # show_progress writes to stdout, which would interleave with the ingest
        # report; page_chunks is what preserves page numbers.
        pages = pymupdf4llm.to_markdown(
            doc, page_chunks=True, show_progress=False
        )
    except MarkdownConversionError:
        raise
    except Exception as exc:
        raise MarkdownConversionError(
            f"Markdown conversion failed for {path.name}: {exc}"
        ) from exc
    finally:
        doc.close()

    per_page: list[list[str]] = []
    lines: list[tuple[str, int]] = []
    for chunk in pages:
        page_no = int((chunk.get("metadata") or {}).get("page") or 0)
        page_lines = [
            _clean_markdown_line(raw) for raw in (chunk.get("text") or "").split("\n")
        ]
        per_page.append([line for line in page_lines if line.strip()])
        lines.extend((line, page_no) for line in page_lines)

    return _drop_running_lines(lines, per_page), page_count


def _drop_running_lines(
    lines: list[tuple[str, int]], per_page: list[list[str]]
) -> list[tuple[str, int]]:
    """Remove running headers/footers, identified by cross-page repetition.

    The markdown converter's ``margins`` setting clips by geometry, which misses a
    header that sits inside the text area -- so the same frequency rule the
    plain-text path uses (``ingestion._strip_running_lines``) is applied here too.
    Without it every chunk in a 22-page handbook carries a copy of the document's
    running title, and the corpus measured ~15% larger than the plain-text path
    purely in repeated chrome.

    Headings and table rows are never dropped, however often they repeat: an H1 is
    usually the same string as the running header, and it is the one line that
    defines the whole document's heading path.
    """
    running = _strip_running_lines(per_page, config.HEADER_FOOTER_MIN_PAGE_RATIO)
    if not running:
        return lines
    return [
        (line, page)
        for line, page in lines
        if line.strip() not in running
        or _HEADING.match(line)
        or _TABLE_ROW.match(line)
    ]


def markdown_text(lines: list[tuple[str, int]]) -> str:
    """Flatten page-tagged lines back to one markdown document."""
    return "\n".join(line for line, _ in lines)


# ---------------------------------------------------------------------------
# Markdown cache
# ---------------------------------------------------------------------------


def markdown_cache_path(source: str, user_id: str | None) -> Path:
    """Where this owner's cached markdown for ``source`` lives.

    Namespaced by owner. Two users may both have uploaded ``handbook.pdf``, and a
    shared file would let one read text extracted from the other's document --
    per-user isolation has to hold for the markdown copy too, not only for the
    vectors.

    The PDF's full filename is kept and ``.md`` appended (``handbook.pdf.md``) so
    a cache entry can never collide with a genuine ``handbook.md``.
    """
    owner = user_id or UNSCOPED_DIR_NAME
    return config.MARKDOWN_DIR / owner / f"{Path(source).name}.md"


def save_markdown(source: str, text: str, user_id: str | None) -> str:
    """Write generated markdown to the cache. Returns a PROJECT_ROOT-relative path.

    A write failure is not fatal: the cache exists so re-chunking experiments and
    table inspection avoid a re-parse, and losing it costs a re-parse, not a
    document.
    """
    path = markdown_cache_path(source, user_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        logger.warning("Could not cache markdown at %s", path, exc_info=True)
        return ""
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:  # MARKDOWN_DIR pointed outside the project
        return str(path)


def load_markdown(source: str, user_id: str | None) -> str:
    """Read cached markdown for ``source``, or "" when there is none."""
    path = markdown_cache_path(source, user_id)
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:  # pragma: no cover - unreadable cache is just a cache miss
        return ""


# ---------------------------------------------------------------------------
# Stage 1: MarkdownHeaderTextSplitter -> sections with a heading hierarchy
# ---------------------------------------------------------------------------


def _header_levels(depth: int) -> list[tuple[str, str]]:
    """``[("#", "h1"), ("##", "h2"), ...]`` up to ``depth`` levels."""
    depth = max(1, min(6, depth))
    return [("#" * n, f"h{n}") for n in range(1, depth + 1)]


def split_sections(
    lines: list[tuple[str, int]], header_depth: int | None = None
) -> list[Section]:
    """Stage 1. Split markdown on heading boundaries, capturing the hierarchy.

    ``MarkdownHeaderTextSplitter`` returns each section's content plus the
    enclosing headings as metadata (``{"h1": ..., "h2": ...}``), which is exactly
    the hierarchy ``heading_path`` needs. ``strip_headers=False`` keeps the
    heading line inside its section: the heading is the most topical sentence a
    section has, and removing it makes a chunk that opens "This applies to all
    staff" state nothing about what "this" is.

    The splitter decides the boundaries and the hierarchy; each section's BODY is
    then sliced out of the original page-tagged lines, which is what preserves
    both page numbers and blank lines. See the comment below for why.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    depth = header_depth if header_depth is not None else config.MARKDOWN_HEADER_LEVELS
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_header_levels(depth),
        strip_headers=False,
    )
    documents = splitter.split_text(markdown_text(lines))

    # The splitter is used for the BOUNDARIES and the hierarchy only; each
    # section's content is then sliced out of the original page-tagged lines.
    #
    # It cannot be used for the content itself: it discards blank lines. Blank
    # lines are the only record of where one paragraph ends and the next begins,
    # and without them ``_reflow`` merges a whole section into one paragraph --
    # measured on employee_handbook.pdf as 137 lines collapsing to 2 paragraphs,
    # which then reported the section's last three pages as its first.
    boundaries: list[int] = []
    cursor = 0
    for document in documents:
        first = next(
            (line.strip() for line in document.page_content.split("\n") if line.strip()),
            "",
        )
        index = _find(first, lines, cursor) if first else -1
        if index < 0:
            # Unmatched: start this section where the previous one ended, so a
            # single normalisation quirk cannot drop a section's whole body.
            index = boundaries[-1] if boundaries else 0
        boundaries.append(index)
        cursor = index + 1

    sections: list[Section] = []
    for position, document in enumerate(documents):
        begin = boundaries[position]
        end = boundaries[position + 1] if position + 1 < len(boundaries) else len(lines)
        headings = [
            str(document.metadata[f"h{n}"]).strip()
            for n in range(1, depth + 1)
            if document.metadata.get(f"h{n}")
        ]
        body = [(line.strip(), page) for line, page in lines[begin:end]]
        while body and not body[-1][0]:
            body.pop()
        if not any(line for line, _ in body):
            continue
        sections.append(
            Section(
                heading_path=" > ".join(headings),
                title=headings[-1] if headings else "",
                lines=body,
            )
        )
    return sections


def _find(text: str, lines: list[tuple[str, int]], cursor: int) -> int:
    """Index of the first line at or after ``cursor`` equal to ``text``, or -1.

    Forward-only: a later duplicate line must not drag a section boundary
    backwards past a section that has already been placed.
    """
    for index in range(cursor, len(lines)):
        if lines[index][0].strip() == text:
            return index
    return -1


def _locate(
    text: str, lines: list[tuple[str, int]], cursor: int, partial: bool = False
) -> tuple[int, int]:
    """Find ``text`` in ``lines`` at or after ``cursor``; return ``(page, cursor)``.

    Forward-only. A line that cannot be matched (whitespace the splitter
    normalised differently) inherits the current cursor's page rather than
    restarting the scan, which would let a later duplicate line -- a repeated
    table cell, say -- drag the page number backwards.

    ``partial`` matches a substring instead of the whole line. Stage 2 needs it:
    the character splitter cuts mid-paragraph, so a chunk's first line is usually
    a *fragment* of a paragraph rather than the whole thing.
    """
    for index in range(cursor, len(lines)):
        candidate = lines[index][0].strip()
        if candidate == text or (partial and text and text in candidate):
            return lines[index][1], index + 1
    fallback = min(cursor, len(lines) - 1) if lines else 0
    return (lines[fallback][1] if lines else 1), cursor


# ---------------------------------------------------------------------------
# Stage 2: oversized sections -> RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------


def _table_runs(section: Section) -> list[tuple[str, list[tuple[str, int]]]]:
    """Partition a section's lines into alternating prose and table runs.

    A markdown table is emitted as its own unit so the character splitter never
    sees it. Half a table has lost its header row and is uninterpretable -- the
    same reason the V2 path keeps PyMuPDF tables whole.
    """
    runs: list[tuple[str, list[tuple[str, int]]]] = []
    for line, page in section.lines:
        kind = "table" if _TABLE_ROW.match(line) else "prose"
        if runs and runs[-1][0] == kind:
            runs[-1][1].append((line, page))
        else:
            runs.append((kind, [(line, page)]))
    return [(kind, run) for kind, run in runs if any(line for line, _ in run)]


def _reflow(run: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Join wrapped lines back into paragraphs, one entry per paragraph.

    PDF text arrives wrapped at the page's line width, so "The company may audit"
    and "compliance with this section" are separate lines of one sentence. Left
    that way, ``RecursiveCharacterTextSplitter`` finds a "\\n" separator between
    them and splits there -- producing a chunk that starts mid-sentence. Measured
    on ``employee_handbook.pdf``: 54 of 78 chunks began mid-sentence before this
    reflow, 0 after.

    A paragraph ends at a blank line, and a heading or list item always starts its
    own paragraph. The page recorded is the page the paragraph STARTS on, which is
    what a citation should point at.
    """
    paragraphs: list[tuple[str, int]] = []
    buffer: list[str] = []
    start_page = 0

    def flush() -> None:
        nonlocal buffer, start_page
        if buffer:
            paragraphs.append((" ".join(buffer), start_page))
            buffer = []

    for line, page in run:
        if not line:
            flush()
            continue
        if _HEADING.match(line) or _BULLET.match(line):
            flush()
            paragraphs.append((line, page))
            continue
        if not buffer:
            start_page = page
        buffer.append(line)
    flush()
    return paragraphs


def sections_to_blocks(
    sections: list[Section],
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[Block]:
    """Stage 2. Turn sections into a page-numbered ``Block`` stream.

    A section whose whole text fits ``max_tokens`` is passed through untouched by
    any character splitter -- the semantic boundary from stage 1 is the only
    boundary. A section over budget goes through the existing
    ``RecursiveCharacterTextSplitter`` at the locked ~800/120 config, and that is
    the only place an arbitrary character boundary can appear. Tables are never
    split.

    Blocks are emitted one per PARAGRAPH rather than one per section, even when
    the section is kept whole. That is a citation requirement, not a chunking
    choice: a ``Block`` carries a single page number, so collapsing a section that
    spans pages 12-15 into one block would report every chunk in it as page 12 and
    strand pages 13-15 -- measured as an 8-page corpus reporting a maximum page of
    19 against the plain-text path's 22. ``ingestion._chunk_blocks`` merges these
    paragraph blocks back up to the child/parent budget and derives the true
    ``page``/``page_end`` range from them, so the boundaries the reader sees are
    still section-then-budget, and they never fall mid-paragraph.
    """
    budget = max_tokens or config.MARKDOWN_SECTION_MAX_TOKENS or config.CHUNK_SIZE
    step = overlap if overlap is not None else config.CHUNK_OVERLAP

    blocks: list[Block] = []
    for section in sections:
        def emit(kind: str, text: str, page: int) -> None:
            text = text.strip()
            if kind == "heading":
                # Keep the heading TEXT in the chunk body -- it is the most
                # topical line the section has, and BM25 needs it verbatim -- but
                # drop the "##" markers so a citation snippet reads as prose.
                text = _HEADING_MARKERS.sub("", text).strip() or text
            if text:
                blocks.append(
                    Block(
                        kind=kind,
                        text=text,
                        page=page,
                        section=section.title,
                        heading_path=section.heading_path,
                    )
                )

        for kind, run in _table_runs(section):
            if kind == "table":
                if config.TABLE_AWARE_INGESTION_ENABLED:
                    # V3.2 owns tables: they are extracted structurally with
                    # PyMuPDF, stitched across page breaks, and chunked by row.
                    # Emitting the markdown rows here as well would index every
                    # table twice, with only one copy stitched and headered.
                    continue
                emit(
                    "table",
                    "\n".join(line for line, _ in run if line),
                    min(page for line, page in run if line),
                )
                continue

            paragraphs = _reflow(run)
            if not paragraphs:
                continue

            # Stage 2 reaches only what is still over budget. A section whose
            # whole text fits cannot contain a paragraph that does not, so such a
            # section passes through with no character splitting at all -- the
            # guarantee the flag is here to make.
            for para, page in paragraphs:
                kind = "heading" if _HEADING.match(para) else "paragraph"
                if count_tokens(para) <= budget:
                    emit(kind, para, page)
                    continue
                for piece in _splitter(budget, step).split_text(para):
                    emit(kind, piece, page)
    return blocks


def oversized_sections(
    sections: list[Section], max_tokens: int | None = None
) -> list[tuple[str, int]]:
    """``[(heading_path, tokens), ...]`` for sections stage 2 had to split.

    Reported by the verification script: "how often did the character splitter
    have to run at all" is the honest measure of how much of the corpus V3.1
    actually splits semantically.
    """
    budget = max_tokens or config.MARKDOWN_SECTION_MAX_TOKENS or config.CHUNK_SIZE
    out = []
    for section in sections:
        tokens = count_tokens(section.text)
        if tokens > budget:
            out.append((section.heading_path, tokens))
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _document_title(sections: list[Section], path: Path) -> str:
    """The document's H1 if it has one, else a readable form of the filename."""
    for section in sections:
        if section.heading_path:
            return section.heading_path.split(">")[0].strip()
    return path.stem.replace("_", " ").title()


def _pages_from_blocks(blocks: list[Block]) -> list[PageText]:
    """Rebuild the per-page text view the plain-text path also exposes."""
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block.text)
    return [
        PageText(page=page, text="\n".join(texts).strip())
        for page, texts in sorted(by_page.items())
        if "\n".join(texts).strip()
    ]


def extract_document_markdown(
    pdf_path: str | Path, user_id: str | None = None
) -> ParsedDocument:
    """Parse a PDF via markdown into the standard ``ParsedDocument`` shape.

    Raises:
        MarkdownConversionError: on conversion failure or a near-empty result.
    """
    path = _require_pdf_path(pdf_path)
    lines, page_count = convert_to_markdown(path)
    full = markdown_text(lines)

    if len(full.strip()) < config.MARKDOWN_MIN_CHARS:
        raise MarkdownConversionError(
            f"Markdown for {path.name} is near-empty "
            f"({len(full.strip())} chars); likely a scanned or image-only PDF."
        )

    sections = split_sections(lines)
    blocks = sections_to_blocks(sections)
    if not blocks:
        raise MarkdownConversionError(f"No usable blocks from markdown for {path.name}.")

    title = _document_title(sections, path)
    for block in blocks:
        if not block.heading_path:
            # Not every PDF has font-based structure to read: `India Relocation
            # Policy Ver1.0.pdf` converts to 134 lines and zero headings, and the
            # preamble above a document's H1 has no enclosing heading either.
            # Rooting those at the document title keeps a citation able to name
            # something. This is provenance, not invention -- the title is the
            # document's own H1 or its filename, and it is already what
            # `doc_title` and the embedded breadcrumb carry. `section_title` is
            # left empty on purpose: there genuinely is no section here, and
            # filling it in would assert a structure the document does not have.
            block.heading_path = title

    # V3.2: tables come from PyMuPDF's structured finder in BOTH ingestion modes,
    # so a table is stitched and headered identically whichever text path is on.
    structured_tables = []
    if config.TABLE_AWARE_INGESTION_ENABLED:
        from backend.tables import extract_tables, tables_hash

        structured_tables = extract_tables(path, path.name)

    cached = save_markdown(path.name, full, user_id)
    return ParsedDocument(
        source=path.name,
        title=title,
        page_count=page_count,
        pages=_pages_from_blocks(blocks),
        blocks=blocks,
        # Hashed over the markdown, not the plain text: flipping the flag changes
        # the chunk stream, so it MUST invalidate the "unchanged, skipped" check.
        # A hash that ignored extraction mode would leave a corpus indexed under
        # the old path and silently report success.
        # V3.2 folds the tables in for the same reason: with table-aware ingestion
        # on, the same markdown yields a different chunk stream. Only when the
        # document has tables -- a table-free one is byte-identical either way.
        content_hash=hashlib.sha256(
            (
                "markdown\n"
                + (f"tables\n{tables_hash(structured_tables)}\n" if structured_tables else "")
                + full
            ).encode("utf-8")
        ).hexdigest(),
        extraction_mode="markdown",
        markdown_path=cached,
        tables=structured_tables,
    )


def extract_document_auto(
    pdf_path: str | Path, user_id: str | None = None
) -> ParsedDocument:
    """Parse a PDF by the configured path, falling back to plain text.

    The fallback chain, in order:

    1. Flag off -> plain text. Nothing else runs, so flag-OFF ingestion is the V2
       pipeline exactly.
    2. Markdown conversion raises, or yields near-empty text -> plain text.
    3. Markdown yields materially less text than the plain-text extractor
       (< ``MARKDOWN_MIN_TEXT_RATIO``) -> plain text. This catches the case the
       character-count check cannot: a PDF whose text layer the markdown
       converter partially loses. Losing content is a correctness failure -- an
       unanswerable question -- so the more complete extraction wins even though
       it has no headings.

    In every fallback case ``extraction_mode`` stays ``"text"``, which is how a
    degraded document stays identifiable after the fact.
    """
    path = _require_pdf_path(pdf_path)

    if not config.MARKDOWN_INGESTION_ENABLED:
        return extract_document(path)

    try:
        parsed = extract_document_markdown(path, user_id=user_id)
    except (MarkdownConversionError, PDFReadError) as exc:
        logger.info("Markdown ingestion fell back to text for %s: %s", path.name, exc)
        return extract_document(path)

    # extract_document is the authority on how much text the PDF really has; it
    # is also cheap relative to embedding, so paying for it to guard against a
    # silent content loss is worth it.
    try:
        text_parsed = extract_document(path)
    except PDFReadError:
        # Plain text failed but markdown worked -- keep the markdown result.
        return parsed

    plain_chars = len(" ".join(b.text for b in text_parsed.blocks))
    md_chars = len(" ".join(b.text for b in parsed.blocks))
    if plain_chars and md_chars < plain_chars * config.MARKDOWN_MIN_TEXT_RATIO:
        logger.warning(
            "Markdown for %s recovered %d chars vs %d from text; using text path.",
            path.name,
            md_chars,
            plain_chars,
        )
        return text_parsed

    return parsed
