"""PDF ingestion: page-aware extraction, cleaning, and chunking.

Phase 1 scope only. This module turns PDFs in a data directory into
LangChain ``Document`` chunks carrying ``{source, page, chunk_index}``
metadata. It does not embed, store, retrieve, or call an LLM — those are
later phases.

Pipeline: ``extract_pages`` (PyMuPDF, per-page text) -> ``clean_text``
(conservative whitespace/hyphenation normalization) -> ``chunk_pages``
(RecursiveCharacterTextSplitter, tiktoken-length, paragraph -> sentence ->
word separators) -> ``load_and_chunk`` (glues the above together per file
or per directory).

``chunk_index`` semantics: a per-document running index, starting at 0,
that increases monotonically across pages in document order. This keeps
chunks globally orderable within a document even though they originate
from independently-split pages, which is what Phase 2's vector store and
citation rendering will rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend import config as _config

PDF_SUFFIX = ".pdf"


class PDFReadError(Exception):
    """Raised when a PDF cannot be located, opened, or read."""


@dataclass(frozen=True)
class PageText:
    """Extracted, cleaned text for a single PDF page.

    Attributes:
        page: 1-indexed page number within the source document.
        text: Cleaned page text (may be empty if the page had no content).
    """

    page: int
    text: str


def _require_pdf_path(pdf_path: str | Path) -> Path:
    """Validate that ``pdf_path`` points to an existing, readable PDF file.

    Raises:
        PDFReadError: if the path does not exist, is not a file, or does
            not have a ``.pdf`` suffix.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise PDFReadError(f"File not found: {path}")
    if not path.is_file():
        raise PDFReadError(f"Not a file: {path}")
    if path.suffix.lower() != PDF_SUFFIX:
        raise PDFReadError(f"Not a PDF (expected .pdf suffix): {path}")
    return path


def clean_text(raw_text: str) -> str:
    """Conservatively clean text extracted from a single PDF page.

    Applies only normalizations that are safe across arbitrary company
    documents (HR policies, manuals, onboarding guides, SOPs):

    - De-hyphenate words broken across a line wrap (e.g. "employ-\\nee"
      becomes "employee").
    - Join line-wrapped sentences within a paragraph into a single line,
      while preserving blank lines as paragraph breaks and preserving
      lines that look like bullets/headings (so structure survives).
    - Collapse runs of horizontal whitespace to a single space.
    - Strip leading/trailing whitespace.

    This intentionally does NOT attempt to detect/remove running headers
    or footers by content matching (e.g. "Page X of Y", company name on
    every page) beyond whitespace/isolated-page-number lines, since
    aggressive pattern removal risks deleting real content. Page-number-
    only lines are dropped as a safe, common case.

    Args:
        raw_text: Text extracted from one PDF page.

    Returns:
        Cleaned text, or an empty string if nothing meaningful remains.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # De-hyphenate line-wrapped words: "informa-\ntion" -> "information".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are just a page number (a common footer/header
        # artifact) or fully empty.
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        cleaned_lines.append(stripped)

    # Re-join into paragraphs: a blank line stays a paragraph break, a
    # bullet/heading-like line starts its own line, and other consecutive
    # non-blank lines (mid-paragraph wraps) are joined with a space.
    paragraphs: list[str] = []
    current: list[str] = []
    bullet_re = re.compile(r"^([-•*•]|\d+[.)]|[A-Za-z][.)])\s+")

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in cleaned_lines:
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
    # Collapse repeated horizontal whitespace left over from extraction.
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    return joined.strip()


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract cleaned, page-numbered text from a single PDF.

    Args:
        pdf_path: Path to a ``.pdf`` file.

    Returns:
        A list of ``PageText`` entries, one per non-empty page, in page
        order. Pages whose cleaned text is empty (e.g. scanned images
        with no extractable text layer, or blank pages) are dropped.

    Raises:
        PDFReadError: if the file is missing, not a PDF, encrypted
            without a usable password, or otherwise unreadable.
    """
    path = _require_pdf_path(pdf_path)

    try:
        doc = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise PDFReadError(f"Could not open PDF {path.name}: {exc}") from exc

    try:
        if doc.is_encrypted:
            # Try an empty password (some "encrypted" PDFs only restrict
            # permissions and open fine with a blank password).
            if not doc.authenticate(""):
                raise PDFReadError(
                    f"PDF {path.name} is encrypted and could not be opened."
                )

        pages: list[PageText] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            raw_text = page.get_text("text")
            text = clean_text(raw_text)
            if text:
                pages.append(PageText(page=index + 1, text=text))
        return pages
    except PDFReadError:
        raise
    except Exception as exc:
        raise PDFReadError(f"Failed reading pages of {path.name}: {exc}") from exc
    finally:
        doc.close()


def _make_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """Build the configured splitter.

    Chunk size/overlap default to ``backend.config.CHUNK_SIZE`` /
    ``CHUNK_OVERLAP`` (backed by ``.env``), read dynamically off the
    ``backend.config`` module (not imported by value) so that callers --
    notably Phase 2's chunk-size tuning eval -- can override them per call
    without needing to reload this module. Length is measured in tokens
    via tiktoken so CHUNK_SIZE means "~N tokens", matching the design
    decision recorded in DECISIONS.md.

    Args:
        chunk_size: override for ``config.CHUNK_SIZE`` (tokens).
        chunk_overlap: override for ``config.CHUNK_OVERLAP`` (tokens).
    """
    size = chunk_size if chunk_size is not None else _config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else _config.CHUNK_OVERLAP
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(
    pages: list[PageText],
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split extracted pages into metadata-tagged chunks.

    Args:
        pages: Page-ordered, cleaned text as returned by ``extract_pages``.
        source: Filename (not a full path) to attach to every chunk's
            metadata as ``source``.
        chunk_size: optional override for ``config.CHUNK_SIZE`` (tokens);
            used by the chunk-size tuning eval to index the same corpus at
            several sizes without mutating global config.
        chunk_overlap: optional override for ``config.CHUNK_OVERLAP``.

    Returns:
        A list of ``Document`` objects, each with ``page_content`` set to
        the chunk text and ``metadata`` set to
        ``{"source": source, "page": <int>, "chunk_index": <int>}``.
        ``chunk_index`` is a per-document running index (see module
        docstring), so it increases monotonically across pages. Chunks
        that are empty/whitespace-only after splitting are dropped.
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    running_index = 0

    for page in pages:
        pieces = splitter.split_text(page.text)
        for piece in pieces:
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
    """Extract, clean, and chunk a single PDF end to end.

    Args:
        pdf_path: Path to a ``.pdf`` file.
        chunk_size: optional override for ``config.CHUNK_SIZE`` (tokens).
        chunk_overlap: optional override for ``config.CHUNK_OVERLAP``.

    Returns:
        Metadata-tagged chunks for that document (see ``chunk_pages``).

    Raises:
        PDFReadError: propagated from ``extract_pages`` for missing,
            non-PDF, or unreadable input.
    """
    path = _require_pdf_path(pdf_path)
    pages = extract_pages(path)
    return chunk_pages(
        pages, source=path.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def load_and_chunk_directory(
    data_dir: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Extract, clean, and chunk every PDF in a directory.

    Args:
        data_dir: Directory containing one or more ``.pdf`` files.
            Non-PDF files are ignored; subdirectories are not traversed.
        chunk_size: optional override for ``config.CHUNK_SIZE`` (tokens);
            used by the chunk-size tuning eval to index the same corpus at
            several sizes without mutating global config.
        chunk_overlap: optional override for ``config.CHUNK_OVERLAP``.

    Returns:
        The concatenation of ``load_and_chunk`` results for each PDF,
        sorted by filename for deterministic ordering. ``chunk_index``
        remains per-document (restarts at 0 for each source file).

    Raises:
        PDFReadError: if ``data_dir`` does not exist, is not a directory,
            or contains no PDF files.
    """
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

    all_chunks: list[Document] = []
    for pdf_path in pdf_paths:
        all_chunks.extend(
            load_and_chunk(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return all_chunks
