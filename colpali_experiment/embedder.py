"""ColQwen2 loading and embedding.

Verified against the illuin-tech/colpali repo and PyPI at experiment time:
``colpali-engine`` 0.3.x, checkpoint ``vidore/colqwen2-v1.0`` (Apache-2.0, top
ViDoRe score among stable checkpoints). Mac MPS needs torch==2.5.1 -- later
2.6.x builds are reported broken for ColQwen2 on MPS -- so that is what this
project's venv has pinned (colpali_experiment only; does not touch the
hybrid pipeline's pinned versions in requirements.txt).
"""

from __future__ import annotations

import threading
from functools import lru_cache

import torch
from PIL import Image

from colpali_experiment import config

# FastAPI runs sync endpoints in a thread pool (backend/main.py's /ask is a
# sync `def`), and the eval harness fires several questions concurrently
# (eval.eval_tool.config.CONCURRENCY). @lru_cache alone does not make the
# FIRST call to a cached function thread-safe -- it only memoizes the
# *return value*, so N concurrent threads that all miss the empty cache all
# enter the function body together. That is exactly what surfaced here:
# multiple threads calling ColQwen2.from_pretrained(..., device_map=device)
# at once corrupted accelerate's meta-device dispatch, raising
# "NotImplementedError: Cannot copy out of meta tensor; no data!" on every
# request during a concurrent eval run (measured, not assumed -- reproduced
# with eval.eval_tool.run_eval --backend colpali, CONCURRENCY=4). A plain
# lock around the load serializes only the one-time ~2GB model load; every
# call after the first returns instantly via lru_cache without ever
# acquiring it.
_load_lock = threading.Lock()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_transformers_chat_template_bug() -> None:
    """Work around a transformers 4.53.x bug, scoped to this module only.

    ``transformers.utils.hub.list_repo_templates`` returns template names
    WITH their ``.jinja`` extension (it only strips the
    ``additional_chat_templates/`` prefix), but
    ``processing_utils.get_processor_dict`` then appends ``.jinja`` a second
    time when building the path to fetch -- producing
    ``additional_chat_templates/sentence_transformers.jinja.jinja``, which does
    not exist on the Hub. ``cached_file`` returns ``None`` for that path, and
    the processor then calls ``open(None, ...)``, crashing every checkpoint
    that ships a ``additional_chat_templates/`` folder (``vidore/colqwen2-v1.0``
    included) under this transformers version -- verified by reading the
    installed library source, not assumed.

    Fixed here by stripping a trailing ``.jinja`` from each returned name
    before it round-trips through that dict, so the rest of transformers
    (which colpali-engine's version pin, and the rest of this project's
    Chroma/BM25 stack, otherwise needs unmodified) is untouched.
    """
    from transformers.utils import hub as _hub

    if getattr(_hub.list_repo_templates, "_smartdoc_colpali_patched", False):
        return
    original = _hub.list_repo_templates

    def _patched(*args, **kwargs):
        return [name[: -len(".jinja")] if name.endswith(".jinja") else name
                for name in original(*args, **kwargs)]

    _patched._smartdoc_colpali_patched = True
    _hub.list_repo_templates = _patched
    # processing_utils imported the name directly, so the module-level
    # reference there needs patching too, not just the defining module's.
    import transformers.processing_utils as _pu

    _pu.list_repo_templates = _patched


@lru_cache(maxsize=1)
def _load_uncached():
    """The actual one-time load. Never call directly -- go through _load()."""
    _patch_transformers_chat_template_bug()
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = _resolve_device(config.COLPALI_DEVICE)
    # bfloat16 has patchy MPS kernel coverage as of torch 2.5.1; float32 is the
    # safe choice off CUDA. CUDA keeps bfloat16 for the memory/speed win.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model = ColQwen2.from_pretrained(
        config.COLPALI_MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(config.COLPALI_MODEL_NAME)
    return model, processor, device


def _load():
    """Load model + processor once per process. Cached: this is a ~2GB model.

    Locked so concurrent callers (see the module docstring above) block on
    the first, real load instead of racing into it -- every call after the
    first returns from lru_cache's memo without blocking on the lock at all,
    since acquiring an uncontended threading.Lock is cheap.
    """
    with _load_lock:
        return _load_uncached()


def device_in_use() -> str:
    return _load()[2]


def embed_page_images(images: list[Image.Image]) -> list[torch.Tensor]:
    """Multi-vector patch embeddings, one tensor per image: ``(n_patches, dim)``.

    Returned on CPU so callers can serialize/store without holding device
    memory, and so the caller does not need to know which device was used.
    """
    model, processor, device = _load()
    batch = processor.process_images(images).to(device)
    with torch.no_grad():
        embeddings = model(**batch)
    return [t.to("cpu") for t in embeddings]


def embed_queries(queries: list[str]) -> list[torch.Tensor]:
    """Multi-vector patch embeddings for text queries, same shape family as
    :func:`embed_page_images` so MaxSim scoring can compare them directly.
    """
    model, processor, device = _load()
    batch = processor.process_queries(queries).to(device)
    with torch.no_grad():
        embeddings = model(**batch)
    return [t.to("cpu") for t in embeddings]


def score(query_embeddings: list[torch.Tensor], page_embeddings: list[torch.Tensor]):
    """Late-interaction MaxSim score matrix: ``(n_queries, n_pages)``.

    ``score_multi_vector`` is a ``@staticmethod`` -- pure tensor arithmetic,
    no model weights involved -- so this deliberately does NOT call
    :func:`_load`. Loading the ~2GB model just to score two already-computed
    embedding tensors would be pure waste, and on resource-constrained
    hardware (no dedicated GPU) that waste is the difference between a cheap
    re-score and an unnecessary multi-second/memory-heavy model load.
    """
    from colpali_engine.models import ColQwen2Processor

    return ColQwen2Processor.score_multi_vector(query_embeddings, page_embeddings)
