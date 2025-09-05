# Document Portal

A FastAPI-based Document Portal for uploading, analyzing, comparing, and chatting with documents using Retrieval-Augmented Generation (RAG). The project supports multiple document formats, multimodal retrieval (text, tables, images) via Unstructured.io + LangChain MultiVector, and includes authentication, pre-commit testing, and CI.

> See the companion assignment write-up: [Assignment.md](./Assignment.md)

---

## Features

- Multi-document ingestion with FAISS vector index per session
- Chat with your documents (RAG) using LangChain
- Multimodal chat for PDFs with text/table/image extraction (Unstructured.io) and MultiVectorRetriever
- Document analysis (metadata and summarization)
- Document comparison
- Authentication (signup/login) and protected routes
- LangChain InMemory cache to speed up repeated LLM calls
- Tests validated via pre-commit and GitHub Actions CI

---

## Tech Stack

- API: FastAPI
- RAG: LangChain, FAISS
- Multimodal extraction: Unstructured.io (PDF partition), Pillow/OpenCV
- LLMs: OpenAI / Groq / Google (configurable)
- Auth: fastapi-users, SQLAlchemy (SQLite by default)
- PDF parsing: PyMuPDF (fitz)

---

## Project Structure (high-level)

- `api/main.py`: FastAPI app, endpoints, startup, auth wiring
- `src/document_ingestion/`: Ingestion for text and multimodal
  - `data_ingestion.py`: ChatIngestor, DocHandler for many file types
  - `mm_ingestion.py`: Multimodal ingestion using Unstructured + MultiVector
- `src/document_chat/`: RAG logic
  - `retrieval.py`: Conversational RAG (text-only)
  - `multimodal/`: Indexer & retrieval chain for multimodal
- `auth/`: fastapi-users integration and models
- `utils/`: LLM cache, model loader, document ops, config loader
- `templates/` and `static/`: Simple HTML UI
- `tests/`: Unit tests for endpoints, ingestion, analyzer, and flags

---

## Setup

### Prerequisites

- Python 3.10+
- API keys for one or more LLM providers (choose at least one):
  - `OPENAI_API_KEY`
  - `GROQ_API_KEY`
  - `GOOGLE_API_KEY`

### Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

- Environment variables (examples):
  - `LLM_PROVIDER` (default: `openai`) one of: `openai`, `groq`, `google`
  - `FAISS_BASE` (default: `faiss_index`)
  - `UPLOAD_BASE` (default: `data`)
  - `FAISS_INDEX_NAME` (default: `index`)
  - `LLM_CACHE_ENABLED` (default: `true`)
- Additional configuration lives in `config/config.yaml` (models, multimodal settings).
- The app also supports a combined JSON secret in `apikeyliveclass` env var; otherwise reads provider-specific env vars.

---

## Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` for the login page. Use the Signup link to create an account, then login to access `/app`.

Optional Streamlit UI (if used):

```bash
streamlit run streamlit_ui.py
```

---

## Key Endpoints

- Public UI: `/`, `/login`, `/signup`
- Protected example: `/protected`
- App UI (requires auth): `/app`
- Analyze: `POST /analyze` (upload file)
- Compare: `POST /compare` (upload `reference`, `actual`)
- Chat index: `POST /chat/index`
  - Form fields: `files` (upload), `multimodal` (true/false), `use_session_dirs` (true/false), `k`, `chunk_size`, `chunk_overlap`
- Chat query: `POST /chat/query`
  - Form fields: `question`, `session_id` (if `use_session_dirs=true`), `multimodal` (true/false), `k`

### Example cURL

```bash
# Analyze
curl -X POST -F "file=@sample.pdf" http://localhost:8080/analyze

# Build chat index (text-only)
curl -X POST \
  -F "files=@sample.pdf" \
  -F "multimodal=false" \
  -F "use_session_dirs=true" \
  http://localhost:8080/chat/index

# Build chat index (multimodal)
curl -X POST \
  -F "files=@sample.pdf" \
  -F "multimodal=true" \
  -F "use_session_dirs=true" \
  http://localhost:8080/chat/index

# Query chat (text-only)
curl -X POST \
  -F "question=What is this document about?" \
  -F "use_session_dirs=true" \
  -F "session_id=<returned_session_id>" \
  -F "multimodal=false" \
  http://localhost:8080/chat/query

# Query chat (multimodal)
curl -X POST \
  -F "question=Summarize the figures and tables" \
  -F "use_session_dirs=true" \
  -F "session_id=<returned_session_id>" \
  -F "multimodal=true" \
  http://localhost:8080/chat/query
```

---

## Supported File Types

Text and ingestion:

- `.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, `.csv`, `.xlsx`, `.xls`, `.db`, `.sqlite`, `.sqlite3`

Multimodal (current focus):

- PDFs via `unstructured.partition.pdf` (extract texts, tables, and images)

---

## Testing & Pre-commit

```bash
# Run tests
pytest tests/

# Install pre-commit hooks
pre-commit install
# Optionally run on all files once
pre-commit run --all-files
```

- Pre-commit is configured to run `pytest` before commits (`.pre-commit-config.yaml`).

---

## Continuous Integration

- GitHub Actions workflow: `.github/workflows/ci.yml`
  - Unit tests on `push`/`pull_request` to `dev`
  - DeepEval jobs (text-only and multimodal) run in non-blocking mode to report metrics

---

## Notes

- FastAPI lifespan initializes LLM cache and creates auth tables on startup.
- Session-based FAISS indices and assets are stored under `faiss_index/<session>` and `data/<session>`.
- Configure LLM provider and keys via environment variables; see Configuration above.

---

## License

This project is for educational purposes. Review dependencies for their respective licenses.
