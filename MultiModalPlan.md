### MultiModal RAG Integration Plan (PDF first, extensible to other file types)

This plan integrates tables and images into the existing document chat module using a modular architecture. It is based on the working approach in `notebook/mutliModal.ipynb` and the current production code paths (`src/document_ingestion`, `src/document_chat`, `api/main.py`). It keeps the existing FAISS-based flow intact and adds an optional multimodal path that can be enabled per request.

---

### Goals and scope

- **Goal**: Enhance RAG to handle text, tables, and images from PDFs now; design to extend to DOCX, PPTX, etc.
- **Non-breaking**: Default remains text-only. Multimodal can be toggled via API/UI; if dependencies are missing, it degrades gracefully to text-only.

---

### Current state summary (as implemented)

- **Ingestion**: `src/document_ingestion/data_ingestion.py` (`ChatIngestor` → `FaissManager`) saves text chunks to FAISS per session.
- **Retrieval**: `src/document_chat/retrieval.py` (`ConversationalRAG`) loads FAISS, builds LCEL chain, answers queries.
- **API**: `api/main.py` exposes `/chat/index` and `/chat/query` for sessionized FAISS-based chat.
- **Models**: `utils/model_loader.py` provides embeddings (Google Generative AI) and chat LLMs (Google/Groq) via config.
- **Start-up**: Uses FastAPI lifespan for initialization. Keep this intact [[note: see `api/main.py`]].

---

### High-level design for multimodality

1. **Partition PDFs into modalities** using `unstructured.partition.pdf` with `strategy="hi_res"` to extract: text blocks, tables (as text), and images (saved as files).
2. **Summarize per-modality** for retrieval:
   - Text → concise summaries optimized for retrieval.
   - Tables → concise table summaries optimized for retrieval.
   - Images → generate captions/summaries using a vision-capable model (Gemini) and keep the base64 or a file path.
3. **Index using MultiVectorRetriever** (LangChain):
   - Vector store: reuse FAISS (existing infra) to store summary Documents with `metadata={doc_id, modality, source, page}`.
   - Payload store: use `LocalFileStore` (persistent docstore) to map `doc_id` → raw content (text/table string or image path/base64).
4. **Query chain**:
   - Retrieve by summaries (FAISS).
   - Resolve raw payloads via docstore.
   - Build a mixed message: include texts/tables as text; include images as image parts when the final LLM supports vision; otherwise, fall back to image captions.
5. **Session persistence**: store FAISS and the docstore under the existing per-session directories.

---

### Directory and module additions

- `src/document_chat/multimodal/`
  - `indexer.py`: PDF parsing, per-modality summarization, and multi-vector retriever builder.
  - `retrieval.py`: RAG chain that formats multimodal prompts and invokes the LLM.
  - `store.py`: Persistent docstore helpers (backed by `LocalFileStore`), utilities for image encoding.
- Optional utilities (if preferred to keep concerns isolated):
  - `utils/mm_store.py`: Small wrapper around `LocalFileStore` with typed helpers.

---

### API changes (compatible)

- Extend existing endpoints with a toggle, defaulting to false:
  - `/chat/index`: add `multimodal: bool = False`.
  - `/chat/query`: add `multimodal: bool = False`.
- Behavior:
  - If `multimodal=False`: current flow (unchanged).
  - If `multimodal=True`: use the multimodal path (index via MultiVectorRetriever + docstore; query via multimodal chain).
- Alternative (if cleaner): add dedicated routes `/chat/mm/index` and `/chat/mm/query` and keep the current ones unchanged.

---

### Persistence layout (per session)

- FAISS index: `faiss_index/<session>/index.*` (unchanged).
- Multimodal FAISS collection (same index files; keep summaries mixed for text/table/image).
- Docstore: `faiss_index/<session>/mm_store/` (LocalFileStore), keys are `doc_id`.
- Extracted assets: `data/<session>/mm_assets/` (images saved as `.jpg` from unstructured).

---

### Data model and metadata

- Vectorstore Documents (summaries):
  - `page_content`: summary text.
  - `metadata`: `{ "doc_id": <uuid>, "modality": "text"|"table"|"image", "source": <file_path>, "page": <int> }`.
- Docstore payloads:
  - For text/table: the original text/table string.
  - For image: image file path (preferred), and optionally cache base64 on-demand.

---

### Configuration and dependencies

- `requirements.txt` additions:
  - `unstructured` (and `unstructured[pdf]` if available)
  - `pillow`
  - `opencv-python-headless`
  - `pytesseract` (if using OCR)
  - `langchain>=0.3.0`, `langchain-community>=0.3.0`
  - Ensure FAISS remains pinned as in repo
- `config/config.yaml` additions:
  - `multimodal.enabled: true|false` (default false)
  - `multimodal.vision_model: "gemini-1.5-flash"` (used for image captioning and optionally final QA)
  - `multimodal.hires_strategy: true|false` (toggle `hi_res` partitioning)
  - `multimodal.max_images: 8` (cap per index/query to manage cost)

---

### Step-by-step implementation

1. Dependencies and config

- Add the packages listed above to `requirements.txt` (guard import errors where needed).
- Extend `config/config.yaml` with the `multimodal` block. Ensure `ModelLoader` can load a vision-capable LLM when requested.

2. Multimodal indexer (`src/document_chat/multimodal/indexer.py`)

- Implement `partition_pdf_to_modalities(path: str, assets_dir: Path) -> dict[str, list]` using `unstructured.partition.pdf` with `extract_images_in_pdf=True` and `extract_image_block_types=["Image","Table"]`.
- Collect lists: `texts: list[str]`, `tables: list[str]`, `images: list[pathlib.Path]` (saved by Unstructured under `assets_dir`).
- Implement summarization functions (prompts from the notebook, adapted):
  - `summarize_texts(texts: list[str]) -> list[str]`
  - `summarize_tables(tables: list[str]) -> list[str]`
  - `summarize_images(images: list[Path]) -> list[str]` using Gemini (via `ModelLoader`).
- Implement `build_multi_vector_retriever(session_paths, summaries_by_modality, payloads_by_modality)`:
  - Create `doc_id` per payload; write payload to `LocalFileStore` (`mset`).
  - For each summary, add a `Document` with `metadata={doc_id, modality, source, page}` to FAISS via `add_documents`.
  - Return a configured `MultiVectorRetriever` with FAISS and the persistent docstore.

3. Multimodal ingestion (`src/document_ingestion/mm_ingestion.py`)

- Create `MultiModalChatIngestor` mirroring `ChatIngestor` but:
  - Saves uploads to session temp dir unchanged.
  - For PDFs: call the indexer to parse → summarize → index via MultiVector.
  - Persist FAISS + docstore under the session directory.
  - Return a retriever handle (or just ensure the session assets are created; querying will reload).

4. Multimodal retrieval chain (`src/document_chat/multimodal/retrieval.py`)

- Utilities from the notebook:
  - `split_image_text_types(docs)` to detect base64 vs text, optionally resize images.
  - `img_prompt_func(context, question)` that composes messages with both image and text parts.
- `build_multimodal_chain(retriever, model_loader, allow_images=True)`:
  - If `allow_images` and `LLM_PROVIDER=google`, load a vision-capable model from `ModelLoader`.
  - Otherwise, fall back to the standard LLM and omit image parts (use image captions within text).
  - Return an LCEL chain: `{context: retriever | split..., question: passthrough} → prompt → llm → StrOutputParser`.

5. Loader for an existing session (FAISS + docstore)

- Implement `load_multimodal_retriever(session_dir)` that:
  - Loads FAISS from `faiss_index/<session>` using existing embeddings.
  - Re-attaches `LocalFileStore` at `faiss_index/<session>/mm_store` and returns a `MultiVectorRetriever`.

6. API updates (`api/main.py`)

- `/chat/index`:
  - Add `multimodal: bool = Form(False)`; when true, use `MultiModalChatIngestor` to build the mm index.
- `/chat/query`:
  - Add `multimodal: bool = Form(False)`; when true, load the multimodal retriever and chain, otherwise the current `ConversationalRAG`.
- Response shape remains `{answer, session_id, k, engine}`; optionally include `{used_images, used_tables}` for diagnostics.

7. UI updates (`templates/index.html`)

- Add a checkbox “Use multimodal (tables+images)” that posts the new flag in both index and query.

8. Tests

- Unit tests:
  - Partitioning on a small synthetic PDF yields non-empty lists for each modality.
  - Summarizers return same-length outputs as inputs.
  - MultiVector indexing: counts per modality present in FAISS metadata; docstore round-trips payloads.
- API tests:
  - `/chat/index` with `multimodal=true` builds an index.
  - `/chat/query` with `multimodal=true` retrieves and answers a question that requires a table value.
- E2E: Use `data_deep_eval/source1.pdf` and validate a known table value is cited in the answer.

9. Extending beyond PDF

- DOCX: use `unstructured.partition.docx` to extract text/tables/images and reuse the same summarization/indexing path.
- PPTX: `unstructured.partition.pptx` for slides and figures.
- CSV/XLS(X): treat each sheet/table as text blocks; optionally render small previews as images if needed.

10. Performance, cost, and limits

- Cap max images per document and per query (e.g., 4–8) to control context size and cost.
- Prefer storing image paths and only base64-encoding at query time.
- If `hi_res` is heavy, allow `strategy="fast"` via config.

---

### Implementation notes and caveats (post-implementation review)

- Retriever k application:

  - Multimodal retriever now respects `k` at query time by wrapping `MultiVectorRetriever` with `.as_retriever(search_kwargs={"k": k})` in `/chat/query`.
  - Index-time `k` is ignored for MM path; index simply writes summaries + payload mappings.

- Vision vs non-vision LLMs:

  - The chain gates images via a `supports_vision` flag (true when `LLM_PROVIDER=google`). When false, image parts are dropped from prompts; image captions can still exist as text summaries for retrieval.

- Payload storage format:

  - Images are stored as file paths in `LocalFileStore` and base64-encoded at retrieval time to keep storage small.

- Unstructured extraction outputs:

  - We currently scan `assets_dir/*.jpg`. If other formats are produced, extend the glob to include `.png/.webp` accordingly.
  - `hi_res` is costlier; toggle via `multimodal.hires_strategy` in config.

- API/UI flags:

  - `multimodal` boolean added to both `/chat/index` and `/chat/query`, reflected in the UI toggle. Default remains false.

- Error handling & keys:

  - `summarize_images` requires `GOOGLE_API_KEY`. Ensure it’s present even if `LLM_PROVIDER` is not Google.
  - Consider catching vision failures and falling back to a stub caption (future improvement).

- Tests added:
  - Route tests for `multimodal=true` paths with monkeypatched components to avoid heavy dependencies.
  - Future unit tests can validate `partition_pdf_to_modalities` on a small fixture and `split_image_text_types` for file-path payloads.

11. Rollout

- Phase 1: Implement PDF multimodality end-to-end, flag-gated and off by default.
- Phase 2: Enable in dev, add tests and DeepEval probes (informational only).
- Phase 3: Expose toggle in UI; gather feedback; then extend to DOCX/PPTX.

---

### Minimal code skeletons (illustrative)

```python
# src/document_chat/multimodal/indexer.py
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_multi_vector_retriever(vectorstore: FAISS, store: LocalFileStore,
                                 text_summaries, texts,
                                 table_summaries, tables,
                                 image_summaries, image_paths):
    # create doc_ids, add summary docs with metadata, and mset payloads
    # return MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    ...
```

```python
# src/document_chat/multimodal/retrieval.py
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def build_multimodal_chain(retriever, llm):
    chain = ({
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    } | RunnableLambda(img_prompt_func) | llm)
    return chain
```

---

### Acceptance criteria

- With `multimodal=true`, a table-only question from a PDF is answered correctly.
- Images can be incorporated when the configured LLM supports vision; otherwise, answers fall back to image captions.
- Existing text-only chat remains unchanged when `multimodal=false`.
