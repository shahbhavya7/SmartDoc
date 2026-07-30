"""Central configuration, loaded from environment variables / .env.

Every tunable (chunk size, overlap, top-k, model names, store location) lives
here so no module hardcodes them.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chunking / retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Storage
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_DIR", "chroma_store")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "smartdoc")

# Service wiring
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
