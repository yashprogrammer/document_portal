I scanned your repo and mapped the LLM and retrieval flows: RAG is in `src/document_chat/retrieval.py` (LCEL chain, FAISS retriever), ingestion is in `src/document_ingestion/data_ingestion.py` (`ChatIngestor` → `FaissManager`), LLMs are loaded via `utils/model_loader.py`, prompts in `prompt/prompt_library.py`, analyzer/comparator pipelines in `src/document_analyzer/data_analysis.py` and `src/document_compare/document_comparator.py`, and the API surface is in `api/main.py`. Now I’ll propose a concrete DeepEval integration plan tailored to these entry points.

### What to evaluate (by module)

- Conversational RAG (`src/document_chat/retrieval.py`)
  - Metrics: Answer Relevancy, Faithfulness (groundedness), Contextual Relevancy/Precision/Recall, Hallucination.
  - Inputs: user query (+ optional chat history), retrieved contexts, final answer.
- Document Analyzer (`src/document_analyzer/data_analysis.py`)
  - Metrics: Faithfulness of summary/metadata to the source text, Answer Relevancy to the task prompt.
  - Inputs: source document text (reference), structured output (converted to text for scoring).
- Document Comparator (`src/document_compare/document_comparator.py`)
  - Metrics: Faithfulness to both inputs and Answer Relevancy to the comparison task; optional custom metric for page-level correctness.
  - Inputs: both documents’ combined text (reference), LLM comparison output.

### Step-by-step plan

1. Dependencies and setup

- Add dependencies to `requirements.txt`:
  - `deepeval`
  - If you want DeepEval metrics to use OpenAI for scoring, also add `openai` (or pick another supported scoring model).
- Environment
  - Keep existing `GOOGLE_API_KEY`/`GROQ_API_KEY` for your app models.
  - Add `OPENAI_API_KEY` if you use OpenAI as the scoring LLM for DeepEval.
  - Optional: set up Confident AI to track runs (`deepeval login` and `CONFIDENT_API_KEY`).

2. Small RAG helper for retrieving the exact contexts used in-chain

- Add a method to `ConversationalRAG` to expose the rewritten question + top-k retrieved docs so DeepEval can score faithfulness/relevancy on precise context. Minimal, non-breaking edit:

```141:173:src/document_chat/retrieval.py
    def get_retrieved_contexts(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None, k: Optional[int] = None):
        if self.retriever is None:
            raise DocumentPortalException("No retriever available", sys)
        # replicate rewriter used in _build_lcel_chain
        question_rewriter = (
            {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
            | self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )
        rewritten = question_rewriter.invoke({"input": user_input, "chat_history": chat_history or []})
        # optionally override k at eval time
        if k is not None and hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = k
        docs = self.retriever.get_relevant_documents(rewritten)
        return rewritten, docs
```

3. Create eval scaffolding

- Create `eval/` with:
  - `eval/datasets/` to store goldens:
    - `chat.jsonl`: list of {session_id or corpus_id, question, expected_answer (or reference_snippet)}.
    - `analyzer.jsonl`: list of {doc_path, expected_summary_snippets or key fields}.
    - `compare.jsonl`: list of {reference_path, actual_path, expected_changes_snippets (optional)}.
  - `eval/common.py`: shared loaders for datasets and a utility to spin up a temporary FAISS index using `ChatIngestor` if needed.

4. RAG evaluation runner

- Add `eval/run_chat_deepeval.py` that:
  - Ensures an index exists for the test corpus (use `ChatIngestor.built_retriver(...)` with a fixed `session_id`, or point to a prebuilt index in `faiss_index/<session>`).
  - Instantiates `ConversationalRAG`, calls `load_retriever_from_faiss(...)`.
  - For each test case, gets final answer via `invoke()`, and contexts via `get_retrieved_contexts()`.
  - Builds DeepEval test cases with metrics:
    - Answer Relevancy (threshold ~0.6–0.8)
    - Faithfulness (uses retrieved contexts)
    - Contextual Relevancy/Precision/Recall (uses retrieved contexts and question)
    - Optional Hallucination
- Example starter:

```python
# eval/run_chat_deepeval.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric

from src.document_chat.retrieval import ConversationalRAG

def build_metrics():
    # choose a scoring model supported by deepeval (e.g., "gpt-4o-mini" if using OpenAI)
    return [
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
        ContextualRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
    ]

def run_one(rag: ConversationalRAG, question: str, expected: str | None = None, k: int = 5):
    answer = rag.invoke(question, chat_history=[])
    _, docs = rag.get_retrieved_contexts(question, chat_history=[], k=k)
    contexts = [getattr(d, "page_content", str(d)) for d in docs]
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected or "",  # optional if you only use context-based metrics
        context=contexts
    )
    assert_test(test_case, build_metrics())

if __name__ == "__main__":
    # assume index already created; otherwise ingest via ChatIngestor before this
    rag = ConversationalRAG(session_id="eval_session")
    rag.load_retriever_from_faiss(index_path="faiss_index/eval_session", k=5, index_name="index")
    run_one(rag, "What is the main topic of source7?", expected="Topic: ...")
```

5. Analyzer evaluation runner

- Add `eval/run_analyzer_deepeval.py`:
  - Read the document via `DocHandler.read_text(...)` or `utils.document_ops.load_documents()` then concatenate.
  - Call `DocumentAnalyzer.analyze_document(text)`; convert structured result to a concise textual form (e.g., title + summary + key fields) for scoring.
  - Use FaithfulnessMetric to reference the original text; optionally Answer Relevancy with a fixed “task prompt” as the input.
- Example starter:

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from src.document_analyzer.data_analysis import DocumentAnalyzer

def normalize_output(meta: dict) -> str:
    parts = []
    if meta.get("Title"): parts.append(f"Title: {meta['Title']}")
    if meta.get("Summary"): parts.append("Summary: " + " ".join(meta["Summary"]))
    return "\n".join(parts)

def evaluate_one(path: str, expected_snippet: str | None = None):
    from src.document_ingestion.data_ingestion import DocHandler
    dh = DocHandler()
    txt = dh.read_text(path)
    analyzer = DocumentAnalyzer()
    out = analyzer.analyze_document(txt)
    test = LLMTestCase(
        input="Analyze the document and summarize key facts.",
        actual_output=normalize_output(out),
        expected_output=expected_snippet or "",
        context=[txt],
    )
    assert_test(test, [FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")])
```

6. Comparator evaluation runner

- Add `eval/run_compare_deepeval.py`:
  - Build combined text exactly as in runtime (`DocumentComparator.combine_documents()`).
  - Call `DocumentComparatorLLM.compare_documents(combined_text)`, convert the rows to a plain text summary for scoring.
  - Score with FaithfulnessMetric (context is combined input) and Answer Relevancy (“Compare two PDFs page-wise” as input).

7. Pytest integration (optional but recommended)

- Add `tests/deepeval/test_rag_eval.py`, `tests/deepeval/test_analyzer_eval.py`, `tests/deepeval/test_compare_eval.py`.
- Mark as slow (e.g., `@pytest.mark.slow` or `@pytest.mark.deepeval`) and exclude from default CI; run on schedule or manually.
- Example:

```python
# tests/deepeval/test_rag_eval.py
import pytest
from eval.run_chat_deepeval import run_one
from src.document_chat.retrieval import ConversationalRAG

@pytest.mark.deepeval
def test_simple_rag_case():
    rag = ConversationalRAG(session_id="eval_session")
    rag.load_retriever_from_faiss("faiss_index/eval_session", k=5, index_name="index")
    run_one(rag, "What is the title of source2?", expected="...")
```

8. Datasets

- Start small with a few well-known Q/A pairs and corresponding reference snippets so Faithfulness can be computed reliably.
- Place under `eval/datasets/` and load them in the runners; keep them lightweight for CI.

9. CI wiring

- Add `.github/workflows/deepeval.yml` to run only on `workflow_dispatch` or nightly:
  - Install deps
  - Prepare a tiny evaluation corpus (or reuse an existing small one)
  - Run `pytest -m deepeval` or call the runners directly
- Keep thresholds modest initially; tighten as retrieval and prompts improve.

10. Observability and iteration

- Optionally enable Confident AI (`deepeval login`) to track runs and regressions.
- Log per-case artifacts (question, rewritten question, top-k contexts, answer) to `logs/` for debugging alongside your existing logger.

### What this changes in your codebase

- New helper in `src/document_chat/retrieval.py`: `get_retrieved_contexts(...)`.
- New `eval/` directory with runners and datasets.
- Optional new pytest files under `tests/deepeval/`.
- `requirements.txt` gain: `deepeval` (+ scoring model client like `openai` if used).
- Optional GitHub workflow for DeepEval.

### Recommended initial metric thresholds

- RAG: Answer Relevancy 0.7, Faithfulness 0.7, Contextual Relevancy 0.7.
- Analyzer: Faithfulness 0.7 (start), adjust after first run.
- Comparator: Faithfulness 0.65–0.7 (page-level comparison is harder).

If you want, I can generate the `eval/` runners and a tiny starter dataset next, and add the `get_retrieved_contexts` helper to `ConversationalRAG`.

- Added a tailored plan covering RAG, Analyzer, Comparator with precise code touch-points (`ConversationalRAG`, `ChatIngestor`, `DocHandler`, `DocumentAnalyzer`, `DocumentComparatorLLM`) and DeepEval runners, plus CI gating and thresholds.
