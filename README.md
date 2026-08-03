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

Current status: **RAG pipeline implemented.** The backend includes PDF ingestion,
persistent Chroma indexing, intent-aware hybrid retrieval, context assembly,
grounded generation, structural citations, and grounding verification/repair.

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

## Run everything

From the project root:

```bash
./run.sh
```

This starts the FastAPI backend and the Streamlit UI together, waits for
`/health` before bringing up the UI, wires the UI to the API, and stops both on
Ctrl-C. Logs go to `.logs/backend.log` and `.logs/frontend.log`.

| | |
|---|---|
| App | <http://localhost:8501> |
| API docs | <http://127.0.0.1:8000/docs> |

Variations:

```bash
./run.sh --backend-only            # API only, for curl / Swagger work
./run.sh --ui-only                 # UI only, against an API already running
API_PORT=8001 UI_PORT=8503 ./run.sh   # a second instance alongside the first
```

The sections below cover starting each service by hand.

## Run backend

From the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload
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
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Streamlit opens on <http://localhost:8501> and shows the backend status card.
It should read **connected**; if it reads *unreachable*, start the backend
first or point `SMARTDOC_API_URL` (or the legacy `BACKEND_URL`) at the right
host.

> Invoke the tools as `.venv/bin/python -m <module>` rather than through the
> venv's `uvicorn` / `streamlit` console scripts. Those scripts hard-code an
> absolute interpreter path in their shebang, so they break if the project
> directory is ever moved. `run.sh` already does this.
