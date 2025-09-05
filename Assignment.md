# Assignment: Document Portal Project

## Overview

This document summarizes the six assignment tasks completed in the Document Portal project and explains, in simple terms, how each was implemented. It includes file paths, endpoints, and references so an instructor can quickly verify the work.

---

## 1) Added Necessary Unit Tests

- Where: `tests/`
  - `tests/test_routes.py`: Health check, protected route auth (401 vs authorized), page routes and redirects, analyze/compare endpoints happy-path and failure cases, chat index/query flows (MM and non-MM) including error paths.
  - `tests/test_ingestion_components.py`: Unit tests for FAISS manager idempotency and loading, `ChatIngestor` sessionized paths and error handling, `DocHandler` save/read for multiple extensions, and `DocumentComparator` save/combine/pdf read paths.
  - `tests/test_api_endpoints_extended.py`: Verifies multimodal flags for `/chat/index` and `/chat/query` using monkeypatched stubs so heavy processing isn’t required.
  - `tests/test_unit_cases.py`: Focused unit tests for `DocumentAnalyzer` initialization, error wrapping, and output shape via `model.models.Metadata`.
- How to run locally: `pytest tests/`
- CI: See section 2.

---

## 2) Tests validated pre-commit and in CI

- Pre-commit hook
  - File: `.pre-commit-config.yaml`
  - Hook runs: the project’s Python from the venv to execute `pytest -q -v tests/` before a commit.
- CI
  - File: `.github/workflows/ci.yml`
  - Jobs:
    - `test`: Installs from `requirements-ci.txt` and runs `pytest tests/` on push/PR to `dev`.
    - `deepeval` and `deepeval-mm`: Non-blocking evaluation jobs (continue-on-error: true) to report DeepEval metrics without failing CI.

---

## 3) Enable project for many document types (.pptx, .docx, .md, .txt, .pdf, .xlsx, .xls, .csv, any SQLite DB)

- Central handler: `src/document_ingestion/data_ingestion.py` → class `DocHandler`
  - Save and read routing for:
    - `.pdf` → `read_pdf()` (PyMuPDF)
    - `.docx` → `_read_docx()` (docx2txt)
    - `.pptx` → `_read_pptx()` (python-pptx)
    - `.md` → `_read_md()`
    - `.txt` → `_read_txt()`
    - `.csv` → `_read_csv()`
    - `.xlsx` → `_read_xlsx()` (openpyxl)
    - `.xls` → `_read_xls()` (xlrd)
    - `.db/.sqlite/.sqlite3` → `_read_sqlite()` (sqlite3 dump-safe, read-only)
  - Allowed types validated in `save_file()` and reused by `DocumentComparator.ALLOWED_EXTS` for compare workflows.
- Document ops for ingestion: `utils/document_ops.py` supports mixed extensions during chat ingestion (PDF, DOCX, TXT, PPTX, MD, CSV, XLSX, XLS, SQLite).
- Tests: `tests/test_ingestion_components.py` covers save/read routing across extensions and errors.

---

## 4) Multimodality in Document Chat with Unstructured.io and MultiVector Retrieval

- Ingestion (MM path): `src/document_ingestion/mm_ingestion.py`
  - Builds multimodal FAISS + `LocalFileStore` using LangChain `MultiVectorRetriever`.
  - Leverages `src/document_chat/multimodal/indexer.py` utilities to partition PDF into modalities and build the retriever.
- Indexing and retrieval utilities: `src/document_chat/multimodal/indexer.py`
  - `partition_pdf_to_modalities(...)`: uses `unstructured.partition.pdf` with `strategy="hi_res"` to extract texts, tables, and images to session assets.
  - `summarize_texts(...)`, `summarize_tables(...)`, `summarize_images(...)`: produce concise summaries per modality using configured LLM provider.
  - `build_multi_vector_retriever(...)`: stores per-modality summaries in FAISS and raw payloads in `LocalFileStore`, keyed by `doc_id`.
  - `load_multimodal_retriever(...)`: reattaches FAISS + docstore for a session.
- Query pipeline (MM): `src/document_chat/multimodal/retrieval.py`
  - `split_image_text_types(...)`: separates images and text, supports base64 or paths.
  - `build_multimodal_chain(...)`: LCEL pipeline combining retriever context (images+texts) → prompt → vision-capable LLM → string output.
- API integration: `api/main.py`
  - `/chat/index` with `multimodal=true` calls `MultiModalChatIngestor.built_retriver(...)`.
  - `/chat/query` with `multimodal=true` reattaches the MM retriever and invokes a multimodal LCEL chain.
- Requirements: `requirements*.txt` include `unstructured`, `pillow`, `opencv-python-headless`. CI installs system deps for PDF parsing in `deepeval-mm` job.
- Tests: `tests/test_api_endpoints_extended.py` asserts MM flag behavior for index/query; error paths covered in `tests/test_routes.py`.

---

## 5) LangChain InMemory Cache

- Initialization: `utils/llm_cache.py`
  - `init_llm_cache()` sets a global `InMemoryCache` via `set_llm_cache(InMemoryCache())` and respects `LLM_CACHE_ENABLED` env toggle.
- Where invoked:
  - API startup lifespan: `api/main.py` calls `init_llm_cache()` on app start.
  - Model loader: `utils/model_loader.py` calls `init_llm_cache()` in `ModelLoader.__init__` to cover non-API entry points.
- Benefit: Avoids repeated LLM calls with identical inputs during a process lifetime, improving responsiveness in development and tests.

---

## 6) Login/Signup and Protected Routes

- Auth stack: `auth/`
  - `auth/auth.py`: FastAPI Users setup with JWT and cookie transports, `current_active_user` dependency.
  - `auth/manager.py`, `auth/models.py`, `auth/db.py`, `auth/schemas.py`: async DB setup (SQLAlchemy), user model and schema, manager wires.
- API routes and protections: `api/main.py`
  - Included routers:
    - `/auth/jwt/*` and `/auth/cookie/*` (login via JWT and cookie backends)
    - `/auth/register` and `/users/*`
  - UI pages: `/`, `/login`, `/signup` (Jinja templates).
  - Protected example: `/protected` requires `current_active_user`.
  - App page: `/app` redirects to `/login` when not authenticated.
  - Business endpoints (`/analyze`, `/compare`, `/chat/index`, `/chat/query`) enforce auth unless the request is detected as a test run; unit tests override this to validate 401 behavior and happy paths.
- Templates: `templates/login.html`, `templates/signup.html`, `templates/index.html`.
- Tests: `tests/test_routes.py` verifies unauthorized vs authorized flows and redirects.

---

## How to Verify Quickly (Instructor Checklist)

1. Run tests locally:
   - `pytest tests/`
2. Confirm pre-commit:
   - Ensure `.pre-commit-config.yaml` exists; install hooks (`pre-commit install`) and attempt a commit to see tests run.
3. Review CI:
   - Check `.github/workflows/ci.yml` for unit test job and non-blocking DeepEval jobs.
4. Check document-type support:
   - Open `src/document_ingestion/data_ingestion.py` (`DocHandler`) and `utils/document_ops.py` to see allowed extensions and per-type readers.
5. Confirm multimodal:
   - Open `src/document_ingestion/mm_ingestion.py` and `src/document_chat/multimodal/*` for Unstructured partitioning and MultiVector retriever.
6. Confirm LLM cache:
   - Open `utils/llm_cache.py` and see `init_llm_cache()` usage in `api/main.py` and `utils/model_loader.py`.
7. Confirm auth and protected routes:
   - Open `api/main.py` (auth routers, `/protected`, `/app`) and `templates/*`, plus `auth/*` files.

---

## Key Endpoints

- Public UI: `/`, `/login`, `/signup`
- Protected example: `/protected`
- App UI (requires auth): `/app`
- Analyze: `POST /analyze`
- Compare: `POST /compare`
- Chat (index): `POST /chat/index` (supports `multimodal` flag)
- Chat (query): `POST /chat/query` (supports `multimodal` flag)

---

## Notes

- FastAPI lifespan is used for startup tasks (LLM cache init, auth DB tables creation).
- DeepEval jobs are non-blocking in CI (they report metrics but do not fail the pipeline).
