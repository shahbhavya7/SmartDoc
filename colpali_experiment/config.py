"""Config for the ColPali experiment. Reads its own env vars, never the
hybrid pipeline's ``backend.config`` values -- the two systems must be able
to change independently of each other.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent

# Own SQLite file, separate from smartdoc.db on purpose (isolation rule 1):
# a bug in this experiment's writes can corrupt at most this one file.
COLPALI_DB_PATH = Path(
    os.getenv("COLPALI_DB_PATH", str(EXPERIMENT_DIR / "colpali_store.db"))
)

# Rendered page images cached here, keyed by document_id/page -- so a re-run
# doesn't re-render every page of every document.
PAGE_IMAGE_DIR = Path(
    os.getenv("COLPALI_PAGE_IMAGE_DIR", str(EXPERIMENT_DIR / "page_images"))
)

# vidore/colqwen2-v1.0: Apache-2.0, the current top ViDoRe-leaderboard
# ColVision checkpoint at experiment time (verified against the illuin-tech
# colpali repo, not assumed from training data -- the space moves fast).
COLPALI_MODEL_NAME = os.getenv("COLPALI_MODEL_NAME", "vidore/colqwen2-v1.0")

# Page render resolution. 150 DPI is what pymupdf4llm/most ColPali examples
# use for a scanned-document-quality image without inflating patch count.
RENDER_DPI = int(os.getenv("COLPALI_RENDER_DPI", "150"))

# Device selection deliberately explicit rather than "auto silently pick
# something": experimental scale here is a handful of PDFs, so CPU is a
# legitimate fallback if the caller wants a deterministic-latency device.
COLPALI_DEVICE = os.getenv("COLPALI_DEVICE", "auto")

# --- Visual table-continuity clustering -----------------------------------
# Own OpenAI key/model, read independently of backend.config -- this
# experiment must not depend on the hybrid pipeline's config module for
# anything, including credentials.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Calibrated against real page-to-page MaxSim scores on this project's own
# multi-page table fixture -- see docs/COLPALI_TABLE_CLUSTERING.md for the
# measured distribution this default was picked from, not guessed.
TABLE_CONTINUITY_THRESHOLD = float(os.getenv("COLPALI_TABLE_CONTINUITY_THRESHOLD", "0.75"))
TABLE_AMBIGUITY_MARGIN = float(os.getenv("COLPALI_TABLE_AMBIGUITY_MARGIN", "0.05"))

# Vision-capable, cheap -- used only for the OPTIONAL ambiguous-pair
# confirmation call, never for the primary embedding path.
VISION_CONFIRM_MODEL = os.getenv("COLPALI_VISION_CONFIRM_MODEL", "gpt-4o-mini")
