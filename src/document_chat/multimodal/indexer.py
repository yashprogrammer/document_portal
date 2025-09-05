from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from utils.model_loader import ModelLoader


@dataclass
class SessionPaths:
    faiss_dir: Path
    mm_store_dir: Path
    assets_dir: Path


def create_session_paths(base_faiss: Path, session_id: str, base_upload: Path) -> SessionPaths:
    faiss_dir = Path(base_faiss) / session_id
    mm_store_dir = faiss_dir / "mm_store"
    assets_dir = Path(base_upload) / session_id / "mm_assets"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    mm_store_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    return SessionPaths(faiss_dir=faiss_dir, mm_store_dir=mm_store_dir, assets_dir=assets_dir)


def build_multi_vector_retriever(
    vectorstore: FAISS,
    store: LocalFileStore,
    text_summaries: List[str],
    texts: List[str],
    table_summaries: List[str],
    tables: List[str],
    image_summaries: List[str],
    image_payloads: List[str],  # paths or base64
) -> MultiVectorRetriever:
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    def add_batch(label: str, summaries: List[str], payloads: List[str]):
        n = min(len(summaries), len(payloads))
        if n == 0:
            return
        ids = [str(uuid.uuid4()) for _ in range(n)]
        docs = [Document(page_content=summaries[i], metadata={id_key: ids[i], "modality": label}) for i in range(n)]
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(ids, payloads[:n])))

    add_batch("text", text_summaries, texts)
    add_batch("table", table_summaries, tables)
    add_batch("image", image_summaries, image_payloads)
    return retriever


def load_faiss_from_dir(index_dir: Path, model_loader: ModelLoader) -> FAISS:
    emb = model_loader.load_embeddings()
    return FAISS.load_local(str(index_dir), embeddings=emb, allow_dangerous_deserialization=True)


def load_multimodal_retriever(session_faiss_dir: Path, model_loader: ModelLoader) -> MultiVectorRetriever:
    """
    Reattach FAISS index and the LocalFileStore for multimodal payloads.
    """
    vs = load_faiss_from_dir(session_faiss_dir, model_loader)
    store = LocalFileStore(str(session_faiss_dir / "mm_store"))
    return MultiVectorRetriever(vectorstore=vs, docstore=store, id_key="doc_id")


def load_multimodal_handles(session_faiss_dir: Path, model_loader: ModelLoader) -> Tuple[MultiVectorRetriever, LocalFileStore]:
    """
    Return both the MultiVectorRetriever and its LocalFileStore for a session.

    This is useful for evaluation where we need to resolve raw payloads by doc_id.
    """
    vs = load_faiss_from_dir(session_faiss_dir, model_loader)
    store = LocalFileStore(str(session_faiss_dir / "mm_store"))
    retriever = MultiVectorRetriever(vectorstore=vs, docstore=store, id_key="doc_id")
    return retriever, store


def build_mm_eval_context(question: str, retriever: MultiVectorRetriever, k: int = 5) -> List[str]:
    """
    Build a normalized textual context list for DeepEval from the multimodal retriever.

    - Retrieves top-k summary Documents
    - For each, resolves the raw payload via doc_id from the LocalFileStore
    - Normalizes by modality into short, textual strings suitable for DeepEval context
    """
    try:
        # Retrieve top-k summary docs using underlying vectorstore
        try:
            docs = retriever.vectorstore.similarity_search(question, k=k)
        except Exception:
            # Fallback to retriever API if available
            docs = retriever.get_relevant_documents(question)  # type: ignore[attr-defined]

        out: List[str] = []
        id_key = getattr(retriever, "id_key", "doc_id")
        store = retriever.docstore

        # Collect payloads by doc_id
        doc_ids: List[str] = []
        for d in docs:
            did = d.metadata.get(id_key)
            if isinstance(did, str):
                doc_ids.append(did)
            else:
                doc_ids.append("")

        payloads = []
        try:
            payloads = list(store.mget(doc_ids)) if hasattr(store, "mget") else []  # type: ignore[attr-defined]
        except Exception:
            payloads = []

        # Normalize contexts
        for i, d in enumerate(docs):
            modality = str(d.metadata.get("modality", "unknown"))
            summary = d.page_content or ""
            payload = payloads[i] if i < len(payloads) else None

            payload_str: str = ""
            if isinstance(payload, bytes):
                try:
                    payload_str = payload.decode("utf-8", errors="ignore")
                except Exception:
                    payload_str = ""
            elif isinstance(payload, str):
                payload_str = payload
            else:
                payload_str = ""

            # For images, payload may be base64 or a file path; prefer textual summary
            if modality == "image":
                normalized = f"modality: image; summary: {summary[:500]}"
            elif modality in {"text", "table"}:
                body = payload_str or summary
                normalized = f"modality: {modality}; content: {body[:1000]}"
            else:
                body = payload_str or summary
                normalized = f"modality: {modality}; content: {body[:800]}"

            out.append(normalized)

        # Keep at most k contexts
        return out[:k]
    except Exception:
        return []


# --------------------------- PDF partition & summarization ---------------------------

def partition_pdf_to_modalities(
    pdf_path: Path,
    assets_dir: Path,
    *,
    hires_strategy: bool = True,
    max_images: int = 8,
) -> Dict[str, List]:
    """
    Use unstructured to split PDF into modalities and extract images to assets_dir.
    Returns dict with keys: texts (List[str]), tables (List[str]), images (List[Path]).
    """
    from unstructured.partition.pdf import partition_pdf  # local import to avoid hard dep at import time

    assets_dir.mkdir(parents=True, exist_ok=True)

    strategy = "hi_res" if hires_strategy else "fast"
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=strategy,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=str(assets_dir),
    )

    texts: List[str] = []
    tables: List[str] = []

    for el in elements:
        t = type(el).__name__
        if t in {"NarrativeText", "Paragraph", "Title", "Header", "ListItem"}:
            texts.append(str(el))
        elif t == "Table":
            tables.append(str(el))

    image_paths: List[Path] = []
    # collect extracted images from the output dir (jpg preferred by unstructured)
    for p in sorted(assets_dir.glob("*.jpg")):
        image_paths.append(p)
    if len(image_paths) > max_images:
        image_paths = image_paths[:max_images]

    return {"texts": texts, "tables": tables, "images": image_paths}


def summarize_texts(texts: List[str], model_loader: ModelLoader) -> List[str]:
    if not texts:
        return []
    cfg = getattr(model_loader, "config", {}) or {}
    mm_cfg = cfg.get("multimodal", {})
    provider = (mm_cfg.get("provider") or "google").lower()
    if provider == "openai":
        oa_cfg = cfg.get("llm", {}).get("openai", {})
        model_name = oa_cfg.get("model_name", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, api_key=model_loader.api_key_mgr.get("OPENAI_API_KEY"), temperature=0, max_tokens=1024)
    elif provider == "groq":
        groq_cfg = cfg.get("llm", {}).get("groq", {})
        model_name = groq_cfg.get("model_name", "llama-3.1-8b-instant")
        llm = ChatGroq(model=model_name, api_key=model_loader.api_key_mgr.get("GROQ_API_KEY"), temperature=0)
    else:
        google_cfg = cfg.get("llm", {}).get("google", {})
        model_name = google_cfg.get("model_name", "gemini-2.0-flash")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=model_loader.api_key_mgr.get("GOOGLE_API_KEY"), max_output_tokens=1024, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
You are an assistant tasked with summarizing text for retrieval. These summaries will be embedded and used to retrieve the raw text chunks. Give a concise summary optimized for retrieval.

Text: {element}
        """.strip()
    )
    chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()
    return chain.batch(texts, {"max_concurrency": 5})


def summarize_tables(tables: List[str], model_loader: ModelLoader) -> List[str]:
    if not tables:
        return []
    cfg = getattr(model_loader, "config", {}) or {}
    mm_cfg = cfg.get("multimodal", {})
    provider = (mm_cfg.get("provider") or "google").lower()
    if provider == "openai":
        oa_cfg = cfg.get("llm", {}).get("openai", {})
        model_name = oa_cfg.get("model_name", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, api_key=model_loader.api_key_mgr.get("OPENAI_API_KEY"), temperature=0, max_tokens=1024)
    elif provider == "groq":
        groq_cfg = cfg.get("llm", {}).get("groq", {})
        model_name = groq_cfg.get("model_name", "llama-3.1-8b-instant")
        llm = ChatGroq(model=model_name, api_key=model_loader.api_key_mgr.get("GROQ_API_KEY"), temperature=0)
    else:
        google_cfg = cfg.get("llm", {}).get("google", {})
        model_name = google_cfg.get("model_name", "gemini-2.0-flash")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=model_loader.api_key_mgr.get("GOOGLE_API_KEY"), max_output_tokens=1024, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
You are an assistant tasked with summarizing tables for retrieval. These summaries will be embedded and used to retrieve the raw table elements. Give a concise summary optimized for retrieval.

Table: {element}
        """.strip()
    )
    chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()
    return chain.batch(tables, {"max_concurrency": 5})


def _encode_image_to_b64(path: Path) -> str:
    return path.read_bytes().hex()  # placeholder replaced below


def summarize_images(image_paths: List[Path], model_loader: ModelLoader, *, prompt_text: str | None = None) -> List[str]:
    if not image_paths:
        return []

    # Choose provider for image summaries based on config.multimodal.provider
    cfg = getattr(model_loader, "config", {}) or {}
    mm_cfg = cfg.get("multimodal", {})
    provider = (mm_cfg.get("provider") or "openai").lower()
    model_name = mm_cfg.get("vision_model", "gpt-4o-mini")

    if provider == "openai":
        chat = ChatOpenAI(model=model_name, api_key=model_loader.api_key_mgr.get("OPENAI_API_KEY"), temperature=0, max_tokens=1024)
        def invoke_img(b64: str, prompt: str) -> str:
            msg = chat.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ])
            ])
            return getattr(msg, "content", "")
    elif provider == "groq":
        chat = ChatGroq(model=model_name, api_key=model_loader.api_key_mgr.get("GROQ_API_KEY"), temperature=0)
        def invoke_img(b64: str, prompt: str) -> str:
            # Groq ChatGroq expects messages; we pass a single HumanMessage with both text and image
            msg = chat.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ])
            ])
            return getattr(msg, "content", "")
    else:
        chat = ChatGoogleGenerativeAI(model=model_name, google_api_key=model_loader.api_key_mgr.get("GOOGLE_API_KEY"), max_output_tokens=1024)
        def invoke_img(b64: str, prompt: str) -> str:
            msg = chat.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ])
            ])
            return getattr(msg, "content", "")

    def encode_image(image_path: Path) -> str:
        import base64
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    prompt = (
        prompt_text
        or "You are an assistant tasked with summarizing images for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary optimized for retrieval."
    )

    outputs: List[str] = []
    for p in image_paths:
        b64 = encode_image(p)
        outputs.append(invoke_img(b64, prompt))
    return outputs


