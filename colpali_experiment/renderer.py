"""Render every page of an existing PDF to a PIL image. No OCR, no text
extraction, no chunking -- ColPali reads the page as a picture.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF -- already a locked dependency of the hybrid pipeline
from PIL import Image

from colpali_experiment import config


def _cache_path(document_id: str, page_number: int) -> Path:
    return config.PAGE_IMAGE_DIR / document_id / f"page_{page_number:04d}.png"


def render_pdf_pages(
    pdf_path: Path, document_id: str, dpi: int | None = None
) -> list[tuple[int, Image.Image]]:
    """Render every page of ``pdf_path`` to a PIL image, 1-indexed.

    Cached on disk under ``page_images/<document_id>/page_NNNN.png`` so a
    re-run does not re-render pages already on disk. Returns
    ``[(page_number, image), ...]`` in page order.
    """
    dpi = dpi or config.RENDER_DPI
    out_dir = config.PAGE_IMAGE_DIR / document_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[tuple[int, Image.Image]] = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            cache_path = _cache_path(document_id, i)
            if cache_path.is_file():
                pages.append((i, Image.open(cache_path).convert("RGB")))
                continue
            pixmap = page.get_pixmap(matrix=matrix)
            image = Image.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )
            image.save(cache_path)
            pages.append((i, image))
    finally:
        doc.close()
    return pages
