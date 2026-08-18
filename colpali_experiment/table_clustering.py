"""Visual table-continuity clustering: does page N+1 continue a table from page N?

Deliberately 100% vision-based, per the brief's hard constraint. This module
never imports ``fitz.Page.find_tables``, never runs OCR, and never touches
``backend.ingestion``/``backend.tables`` (the hybrid pipeline's text-based
table stitching, DECISIONS.md T1-T11). Those two mechanisms solve different
problems and are kept structurally separate so this experiment cannot quietly
regain a dependency on text extraction:

* The hybrid pipeline's table stitching decides continuation from PARSED TEXT
  structure (column count, header repetition) via PyMuPDF's table finder.
* This module decides continuation from PIXELS: the same late-interaction
  MaxSim mechanism ColPali already uses for query-to-page scoring, applied
  page-embedding-to-page-embedding instead of query-to-page.

Why MaxSim is a reasonable continuity signal at all: two pages of the same
table share near-identical visual structure -- a repeated header row, the same
column grid, the same font and cell shading -- and ColPali's patch embeddings
are exactly what should fire on that, since capturing fine-grained visual
layout is what the model is trained for. Two unrelated prose pages share
almost none of that, so the score should separate cleanly. That is measured
below (:func:`pairwise_scores`, calibration script), not assumed.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field

import torch
from PIL import Image

from colpali_experiment import config, store
from colpali_experiment.embedder import score as maxsim_score
from colpali_experiment.renderer import render_pdf_pages

logger = logging.getLogger("colpali_experiment.table_clustering")


@dataclass
class PagePairScore:
    page_a: int
    page_b: int
    raw_score: float
    normalized_score: float  # raw_score / n_patches_a -- see pairwise_scores
    is_continuation: bool
    confirmed_by: str = "threshold"  # "threshold" | "vision_llm"


@dataclass
class ClusteringResult:
    document_id: str
    pair_scores: list[PagePairScore] = field(default_factory=list)
    groups: dict[int, str] = field(default_factory=dict)  # page_number -> group id


def pairwise_scores(document_id: str) -> list[PagePairScore]:
    """MaxSim(page_i, page_{i+1}) for every consecutive page pair, using the
    embeddings already stored for this document.

    Normalized by patch count: raw MaxSim under ``score_multi_vector`` sums one
    best-matching-query-patch score per patch of the FIRST argument, so it
    scales with how many patches page A has, not with how visually similar the
    pages are. Dividing by ``n_patches`` of page A turns it into an average
    per-patch agreement, which is comparable across pages of different sizes
    and is the quantity a single threshold can reason about.
    """
    rows = store.get_document_embeddings(document_id)
    if len(rows) < 2:
        return []

    results = []
    for a, b in zip(rows, rows[1:]):
        emb_a = torch.from_numpy(a["embedding"])
        emb_b = torch.from_numpy(b["embedding"])
        # score_multi_vector expects lists of per-item tensors; treat page A as
        # the "query" side and page B as the "document" side, both length 1.
        raw = float(maxsim_score([emb_a], [emb_b])[0][0])
        normalized = raw / a["n_patches"]
        results.append(
            PagePairScore(
                page_a=a["page_number"],
                page_b=b["page_number"],
                raw_score=raw,
                normalized_score=normalized,
                is_continuation=normalized >= config.TABLE_CONTINUITY_THRESHOLD,
            )
        )
    return results


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _confirm_with_vision_llm(page_a_image: Image.Image, page_b_image: Image.Image) -> bool:
    """One image-only chat completion: does page B continue a table from page A?

    No text from either page is extracted or sent -- both images travel as
    ``image_url`` content parts, so this stays inside the brief's "vision
    calls only" constraint. Used only for pairs whose MaxSim score falls
    within AMBIGUITY_MARGIN of the threshold; a clear score never pays for
    this call.
    """
    import openai

    if not config.OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY set; skipping vision-LLM confirmation.")
        return False

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0, max_retries=2)
    response = client.chat.completions.create(
        model=config.VISION_CONFIRM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Image 1 is one page of a document. Image 2 is the next "
                            "page. Does image 2 continue a TABLE that started in "
                            "image 1 (e.g. the same columns continuing with more "
                            "rows, with or without the header repeated)? Answer "
                            "with exactly one word: yes or no."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(page_a_image)}},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(page_b_image)}},
                ],
            }
        ],
        max_tokens=5,
    )
    answer = (response.choices[0].message.content or "").strip().lower()
    return answer.startswith("y")


def cluster_document(
    document_id: str,
    pdf_path=None,
    use_vision_confirmation: bool = False,
) -> ClusteringResult:
    """Assign every stored page of ``document_id`` to a visual table group.

    A run of consecutive pages joined by ``is_continuation`` (or a confirmed
    ambiguous pair) becomes one group, id ``"{document_id}:{first_page}"`` --
    deliberately NOT reusing or resembling the hybrid pipeline's ``table_id``
    scheme (T3/T8 in DECISIONS.md), so the two mechanisms can never be
    confused for each other downstream. A page that continues nothing is its
    own singleton group.

    ``pdf_path`` is required only when ``use_vision_confirmation=True``, to
    re-render the two images an ambiguous pair needs (rendered images are
    already cached by :mod:`colpali_experiment.renderer`, so this costs
    nothing on a repeat run).
    """
    pairs = pairwise_scores(document_id)
    rows = store.get_document_embeddings(document_id)
    if not rows:
        return ClusteringResult(document_id=document_id)

    page_images: dict[int, Image.Image] | None = None
    if use_vision_confirmation:
        if pdf_path is None:
            raise ValueError("pdf_path is required when use_vision_confirmation=True")
        page_images = dict(render_pdf_pages(pdf_path, document_id))

    continues = {}
    for pair in pairs:
        decision = pair.is_continuation
        margin = abs(pair.normalized_score - config.TABLE_CONTINUITY_THRESHOLD)
        if use_vision_confirmation and margin <= config.TABLE_AMBIGUITY_MARGIN:
            confirmed = _confirm_with_vision_llm(
                page_images[pair.page_a], page_images[pair.page_b]
            )
            pair.is_continuation = confirmed
            pair.confirmed_by = "vision_llm"
            decision = confirmed
        continues[(pair.page_a, pair.page_b)] = decision

    groups: dict[int, str] = {}
    page_numbers = sorted(r["page_number"] for r in rows)
    current_group_start = page_numbers[0]
    for i, page in enumerate(page_numbers):
        if i > 0:
            prev = page_numbers[i - 1]
            if not continues.get((prev, page), False):
                current_group_start = page
        groups[page] = f"{document_id}:{current_group_start}"

    for page, group_id in groups.items():
        store.set_table_group(document_id, page, group_id)

    return ClusteringResult(document_id=document_id, pair_scores=pairs, groups=groups)
