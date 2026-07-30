# SmartDoc

## Overview

SmartDoc is a RAG-based document Q&A assistant. Employees ask plain-English
questions and get accurate, cited answers drawn from a library of company PDFs
(HR policies, product manuals, onboarding guides, SOPs).

The system answers **only** from retrieved document context. When the context
does not contain the answer, it replies "I don't know based on the available
documents" rather than guessing. Every answer returns structured sources
(document, page, snippet).

**Stack**

| Layer | Choice |
| --- | --- |
| Backend / all RAG logic | FastAPI (Python) |
| Vector DB | ChromaDB (`PersistentClient`, on disk) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini` (low temperature) |
| Frontend | Streamlit thin client over HTTP |
| Orchestration | LangChain where it helps |

**Layout**

```
smartdoc/
  backend/       # FastAPI app + RAG pipeline modules
  app/           # Streamlit UI
  data/          # test PDFs
  chroma_store/  # persisted vector DB (git-ignored)
  .env           # real secrets (git-ignored)
  .env.example   # documents variable names only
  requirements.txt
  README.md
```

Current status: **Phase 0 — scaffold only.** No ingestion, retrieval, or
generation yet.

## Setup

```bash
git clone <repo-url> smartdoc
cd smartdoc

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then add your real OPENAI_API_KEY
```

Configuration lives entirely in `.env` (see `.env.example` for the full list):
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBED_MODEL`, `CHAT_MODEL`. Never
commit `.env`.

## Run backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

The API listens on <http://127.0.0.1:8000>. Verify it:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

Interactive docs: <http://127.0.0.1:8000/docs>

## Run frontend

In a second terminal, with the virtualenv active and the backend running:

```bash
streamlit run app/streamlit_app.py
```

Streamlit opens on <http://localhost:8501> and shows the backend status card.
It should read **connected**; if it reads *unreachable*, start the backend
first or point `BACKEND_URL` in `.env` at the right host.
