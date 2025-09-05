### DeepEval Integration Plan for Multimodal Document Chat

This plan integrates DeepEval for the multimodal document chat flow, mirroring the existing text-only DeepEval runner (`eval/run_doc_chat_deepeval.py`) and the workflow demonstrated in `notebook/deepEval.ipynb`. No code is implemented in this document; it outlines the exact steps and artifacts to add next.

---

### Scope and alignment

- **Align with existing pattern**: Use the same dataset, metrics, and evaluate() pattern as the current doc chat DeepEval runner.
- **Reuse infra**: Continue to use per-session FAISS storage and the multimodal docstore introduced by the multimodal flow (`src/document_chat/multimodal/*`, `src/document_ingestion/mm_ingestion.py`, API toggles in `api/main.py`).
- **Non-blocking CI**: Keep DeepEval jobs informational only in CI.

---

### Step-by-step plan

1. Dependencies and environment

- Ensure `deepeval` and your scoring model client (e.g., `openai`) are installed; keep `OPENAI_API_KEY` available for scoring and reuse your existing app keys (`GOOGLE_API_KEY`, `GROQ_API_KEY`).
- Keep thresholds consistent with the notebook and text-only runner to start.

2. Multimodal context extraction helper

- Implement a helper to build normalized textual contexts from the multimodal retriever + store for DeepEval scoring:
  - Retrieve top-k multimodal summary Documents from `MultiVectorRetriever`.
  - Resolve underlying raw payloads via `doc_id` in the `LocalFileStore`.
  - Normalize by modality into short strings:
    - text/table → plain text content
    - image → existing image captions/summaries; optionally add a short `image_ref` (filename)
  - Output: `List[str]` to pass as `context` and `retrieval_context`.
- Location: place in `src/document_chat/multimodal/indexer.py` or `src/document_chat/multimodal/retrieval.py` where both retriever and store access is convenient.

3. Loader returning both retriever and store

- Confirm or add a function to reattach the session’s `MultiVectorRetriever` and `LocalFileStore` together (e.g., extend `load_multimodal_retriever` or add `load_multimodal_handles`).
- This mirrors the text-only `get_retrieved_context` convenience in `ConversationalRAG`.

4. New runner: `eval/run_mm_doc_chat_deepeval.py`

- Structure identical to `eval/run_doc_chat_deepeval.py`:
  - Ingest PDFs under `DEEPEVAL_INPUT_DIR` with `MultiModalChatIngestor` to build FAISS + `mm_store`.
  - Pull an `EvaluationDataset` (alias, e.g., `test_doc_chat_mm`) or load a tiny local golden set.
  - For each golden:
    - Reattach mm retriever (+ store), build mm chain (`build_multimodal_chain`).
    - `answer = chain.invoke(question)`.
    - `contexts = get_mm_context(question, retriever, store, k=...)` (from step 2).
    - Build `LLMTestCase` with `context` and `retrieval_context` set to `contexts`.
- Metrics:
  - `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`, optionally `HallucinationMetric`.
- Evaluate via `deepeval.evaluate(test_cases=..., metrics=...)`.

5. Vision-capability handling

- Detect provider/model capability (OpenAI/Groq/Google). If vision is unsupported, drop image payloads and rely on their captions/textual summaries, in line with `build_multimodal_chain` fallback behavior.

6. Tiny evaluation dataset

- Prepare a minimal, deterministic dataset under `data_deep_eval/` with PDFs containing text, tables, and images.
- Create 3–10 goldens targeting:
  - Text-only lookups (control)
  - Table extraction
  - Image-derived facts (caption comprehension)
- Publish as a Confident AI dataset (e.g., alias `test_doc_chat_mm`) or provide a local loader for offline runs.

7. Logging and traceability

- Log (per test case) the session id, selected `doc_id`s, modalities, and a short preview of each normalized context string.
- Store alongside existing logs for parity with the text-only runner.

8. Optional pytest wrapper

- Add `tests/deepeval/test_mm_rag_eval.py` with `@pytest.mark.deepeval` to execute a single golden using runner helpers.
- Exclude from default CI; execute only via a dedicated marker or workflow.

9. CI wiring (non-blocking)

- Extend `.github/workflows/ci.yml` with a second DeepEval job for multimodal:
  - Install `requirements-deepeval.txt`.
  - Run `python eval/run_mm_doc_chat_deepeval.py` with env:
    - `DEEPEVAL_INPUT_DIR=data_deep_eval`
    - `DEEPEVAL_DATASET_ALIAS=test_doc_chat_mm`
  - Use `continue-on-error: true` so results are informational only.

10. Thresholds and reporting

- Start thresholds similar to text-only; optionally relax slightly for early multimodal runs:
  - Answer/Contextual Relevancy, Precision, Recall: ~0.6–0.7
  - Faithfulness: ~0.65–0.7
- Tighten later as mm ingestion, captioning, and retrieval stabilize.

11. Keep notebook parity

- Match the dataset, metrics list, and the `evaluate(...)` call style from `notebook/deepEval.ipynb` for reproducibility.

12. Future refinements (post-initial integration)

- Add modality-aware context capping (e.g., `max_images`) and shorter captions for tabular snippets.
- Consider a lightweight custom post-check to ensure image-derived facts are present when a golden requires them (informational only).

---

### Artifacts to be added (when implementing)

- `src/document_chat/multimodal/`:
  - Helper to build normalized contexts for DeepEval
  - Loader that returns both retriever and store (if not already available)
- `eval/run_mm_doc_chat_deepeval.py` (mirror of the text-only runner)
- `tests/deepeval/test_mm_rag_eval.py` (optional)
- `.github/workflows/ci.yml`: add a non-blocking multimodal DeepEval job
