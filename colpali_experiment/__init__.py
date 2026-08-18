"""ColPali visual-retrieval experiment -- isolated from every V2/V3 path.

Nothing here is imported by ``backend/*``, and nothing here writes to
``smartdoc.db`` or ``chroma_store/``. Storage lives in its own SQLite file
(``colpali_experiment/colpali_store.db``), created on first use. See
``docs/DECISIONS.md`` for the hybrid pipeline this is being compared against.
"""
