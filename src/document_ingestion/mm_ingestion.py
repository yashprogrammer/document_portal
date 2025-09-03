from __future__ import annotations
import os
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from utils.file_io import generate_session_id, save_uploaded_files
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

from src.document_chat.multimodal.indexer import (
    SessionPaths,
    create_session_paths,
    partition_pdf_to_modalities,
    summarize_texts,
    summarize_tables,
    summarize_images,
)


class MultiModalChatIngestor:
    def __init__(self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: str | None = None,
    ):
        try:
            self.model_loader = ModelLoader()
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            sp = create_session_paths(self.faiss_base, self.session_id, self.temp_base)
            self.session_paths: SessionPaths = sp

            cfg = self.model_loader.config.get("multimodal", {})
            self.hires_strategy = bool(cfg.get("hires_strategy", True))
            self.max_images = int(cfg.get("max_images", 8))

            log.info("MultiModalChatIngestor initialized",
                     session_id=self.session_id,
                     faiss_dir=str(self.session_paths.faiss_dir),
                     mm_store=str(self.session_paths.mm_store_dir),
                     assets=str(self.session_paths.assets_dir))
        except Exception as e:
            log.error("Failed to initialize MultiModalChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in MultiModalChatIngestor", sys)

    def _load_or_create_vectorstore(self, docs: List[Document]) -> FAISS:
        emb = self.model_loader.load_embeddings()
        index_dir = self.session_paths.faiss_dir
        index_path_faiss = index_dir / "index.faiss"
        index_path_pkl = index_dir / "index.pkl"

        if index_path_faiss.exists() and index_path_pkl.exists():
            vs = FAISS.load_local(str(index_dir), embeddings=emb, allow_dangerous_deserialization=True)
            if docs:
                vs.add_documents(docs)
                vs.save_local(str(index_dir))
            return vs

        # Create fresh from documents
        if not docs:
            raise DocumentPortalException("No documents to create FAISS index", sys)
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(str(index_dir))
        return vs

    def built_retriver(self, uploaded_files: Iterable, *, k: int = 5):
        try:
            # Save uploads to session temp dir
            paths = save_uploaded_files(uploaded_files, self.session_paths.assets_dir.parent)  # keep under session data dir

            texts_all: List[str] = []
            tables_all: List[str] = []
            images_all: List[Path] = []

            for p in paths:
                if p.suffix.lower() == ".pdf":
                    out = partition_pdf_to_modalities(
                        p,
                        self.session_paths.assets_dir,
                        hires_strategy=self.hires_strategy,
                        max_images=self.max_images,
                    )
                    texts_all.extend(out.get("texts", []))
                    tables_all.extend(out.get("tables", []))
                    images_all.extend(out.get("images", []))
                else:
                    log.warning("Skipping non-PDF for multimodal ingestion", path=str(p))

            # Summarize per modality
            text_summaries = summarize_texts(texts_all, self.model_loader)
            table_summaries = summarize_tables(tables_all, self.model_loader)
            image_summaries = summarize_images(images_all, self.model_loader)

            # Build summary docs and payload mappings
            id_key = "doc_id"
            docs: List[Document] = []
            kv_pairs: List[tuple[str, str]] = []

            for i, s in enumerate(text_summaries):
                doc_id = str(uuid.uuid4())
                docs.append(Document(page_content=s, metadata={id_key: doc_id, "modality": "text"}))
                kv_pairs.append((doc_id, texts_all[i]))

            for i, s in enumerate(table_summaries):
                doc_id = str(uuid.uuid4())
                docs.append(Document(page_content=s, metadata={id_key: doc_id, "modality": "table"}))
                kv_pairs.append((doc_id, tables_all[i]))

            for i, s in enumerate(image_summaries):
                doc_id = str(uuid.uuid4())
                docs.append(Document(page_content=s, metadata={id_key: doc_id, "modality": "image"}))
                kv_pairs.append((doc_id, str(images_all[i])))

            # Vectorstore (FAISS) and store (LocalFileStore)
            vs = self._load_or_create_vectorstore(docs)
            store = LocalFileStore(str(self.session_paths.mm_store_dir))
            if kv_pairs:
                # LocalFileStore expects bytes values; encode UTF-8 for strings/paths
                kv_bytes = []
                for k, v in kv_pairs:
                    if isinstance(v, bytes):
                        kv_bytes.append((k, v))
                    else:
                        kv_bytes.append((k, str(v).encode("utf-8")))
                store.mset(kv_bytes)

            retriever = MultiVectorRetriever(vectorstore=vs, docstore=store, id_key=id_key)
            # Return MultiVectorRetriever; k will be applied at query time via search_kwargs
            retriever.search_kwargs = {"k": k}
            return retriever

        except Exception as e:
            log.error("Failed to build multimodal retriever", error=str(e))
            raise DocumentPortalException("Failed to build multimodal retriever", e) from e


