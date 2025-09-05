import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval import evaluate

from logger import GLOBAL_LOGGER as log

from src.document_ingestion.mm_ingestion import MultiModalChatIngestor
from src.document_chat.multimodal.indexer import (
    load_multimodal_handles,
    build_mm_eval_context,
)
from src.document_chat.multimodal.retrieval import build_multimodal_chain
from utils.model_loader import ModelLoader


DEEPEVAL_INPUT_DIR = os.getenv("DEEPEVAL_INPUT_DIR", "data_deep_eval")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "notebook/eval_data")
FAISS_BASE = os.getenv("FAISS_BASE", "notebook/eval_faiss_index")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
DATASET_ALIAS = os.getenv("DEEPEVAL_DATASET_ALIAS", "test_doc_chat_mm")


class LocalFileAdapter:
    def __init__(self, file_path: str):
        self.name = os.path.basename(file_path)
        self._file_path = file_path

    def read(self) -> bytes:
        with open(self._file_path, "rb") as f:
            return f.read()

    def getbuffer(self) -> bytes:
        return self.read()


def list_supported_files(root: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def query_mm(question: str, session_id: str, k: int = 5) -> dict:
    index_dir = os.path.join(FAISS_BASE, session_id)
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"FAISS index not found at: {index_dir}")

    model_loader = ModelLoader()
    retriever, store = load_multimodal_handles(Path(index_dir), model_loader)
    chain = build_multimodal_chain(retriever, model_loader.load_llm())
    answer = chain.invoke(question)
    context_list = build_mm_eval_context(question, retriever, k=k)
    return {"answer": answer, "context": context_list}


def main():
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()
        log.info("Running in LOCAL mode: .env loaded")

    data_dir = Path(DEEPEVAL_INPUT_DIR)
    assert data_dir.exists(), f"Input dir not found: {data_dir}"
    paths = list_supported_files(data_dir)
    if not paths:
        log.error("No supported files found in input directory", dir=str(data_dir))
        print("No supported files found in input directory.")
        sys.exit(1)

    # Ingest and index multimodal
    mm = MultiModalChatIngestor(temp_base=UPLOAD_BASE, faiss_base=FAISS_BASE)
    adapters = [LocalFileAdapter(str(p)) for p in paths if p.suffix.lower() == ".pdf"]
    if not adapters:
        log.error("No PDFs found for multimodal evaluation", dir=str(data_dir))
        print("No PDFs found for multimodal evaluation.")
        sys.exit(1)
    mm.built_retriver(adapters)
    log.info("Multimodal ingestion complete", session_id=mm.session_id)

    # Pull dataset (with fallback to a tiny local set if none available)
    dataset = EvaluationDataset()
    try:
        dataset.pull(alias=DATASET_ALIAS)
    except Exception as e:
        log.warning("Failed to pull dataset; falling back to local goldens", error=str(e))

    goldens = getattr(dataset, "goldens", []) or []
    if not goldens:
        # Fallback: minimal goldens relying on context-based metrics
        class _G:
            def __init__(self, q: str, exp: str = ""):
                self.input = q
                self.expected_output = exp
        goldens = [
            _G("Summarize the key information from the document(s) including any tables and images."),
            _G("What information can be inferred from the images?"),
        ]

    # Build test cases
    for golden in goldens:
        try:
            result = query_mm(golden.input, session_id=mm.session_id)
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=result["answer"],
                expected_output=getattr(golden, "expected_output", ""),
                retrieval_context=result["context"],
                context=result["context"],
            )
            dataset.add_test_case(test_case)
        except Exception as e:
            log.error("Failed to build mm test case", error=str(e), question=golden.input)

    metrics = [
        AnswerRelevancyMetric(),
        FaithfulnessMetric(),
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
        ContextualRelevancyMetric(),
        HallucinationMetric(),
    ]

    evaluate(test_cases=dataset.test_cases, metrics=metrics)


if __name__ == "__main__":
    main()


