import os
import json
from pathlib import Path
import types
import io
import builtins

import pytest
from langchain_core.documents import Document

from src.document_ingestion.data_ingestion import FaissManager, ChatIngestor, DocHandler, DocumentComparator


def test_faissmanager_no_data_raises(tmp_path, monkeypatch):
    # Ensure index files don't exist
    mgr = FaissManager(tmp_path)

    # Stub embeddings + FAISS to avoid heavy deps
    class FakeFAISS:
        @staticmethod
        def load_local(*args, **kwargs):
            return object()
        @staticmethod
        def from_texts(*args, **kwargs):
            return object()

    monkeypatch.setattr("src.document_ingestion.data_ingestion.FAISS", FakeFAISS)

    with pytest.raises(Exception):
        mgr.load_or_create(texts=None, metadatas=None)


def test_faissmanager_idempotency(tmp_path, monkeypatch):
    added_docs = []

    class FakeVS:
        def __init__(self):
            self.docs = []
        def add_documents(self, docs):
            self.docs.extend(docs)
        def save_local(self, *args, **kwargs):
            pass

    class FakeFAISS:
        @staticmethod
        def load_local(*args, **kwargs):
            return FakeVS()
        @staticmethod
        def from_texts(*args, **kwargs):
            return FakeVS()

    monkeypatch.setattr("src.document_ingestion.data_ingestion.FAISS", FakeFAISS)

    mgr = FaissManager(tmp_path)
    # Create first time
    texts = ["a", "b"]
    metas = [{"source": "s", "row_id": 1}, {"source": "s", "row_id": 2}]
    vs = mgr.load_or_create(texts=texts, metadatas=metas)
    n1 = mgr.add_documents([Document(page_content="a", metadata=metas[0]), Document(page_content="b", metadata=metas[1])])
    assert n1 == 2
    n2 = mgr.add_documents([Document(page_content="a", metadata=metas[0]), Document(page_content="b", metadata=metas[1])])
    assert n2 == 0


def test_faissmanager_load_existing_without_texts(tmp_path, monkeypatch):
    # Create fake existing index files
    (tmp_path / "index.faiss").write_bytes(b"faiss")
    (tmp_path / "index.pkl").write_bytes(b"pkl")

    class FakeVS:
        def add_documents(self, docs):
            pass
    class FakeFAISS:
        @staticmethod
        def load_local(*args, **kwargs):
            return FakeVS()

    monkeypatch.setattr("src.document_ingestion.data_ingestion.FAISS", FakeFAISS)

    mgr = FaissManager(tmp_path)
    vs = mgr.load_or_create(texts=None, metadatas=None)
    assert isinstance(vs, FakeVS)


def test_chatingestor_resolve_dir_sessionized(tmp_path, monkeypatch):
    ing = ChatIngestor(temp_base=str(tmp_path / "data"), faiss_base=str(tmp_path / "faiss"), use_session_dirs=True)
    assert ing.temp_dir.name == ing.session_id
    assert ing.faiss_dir.name == ing.session_id


def test_chatingestor_empty_docs_raises(tmp_path, monkeypatch):
    ing = ChatIngestor(temp_base=str(tmp_path / "data"), faiss_base=str(tmp_path / "faiss"), use_session_dirs=True)

    monkeypatch.setattr("src.document_ingestion.data_ingestion.save_uploaded_files", lambda uploaded_files, d: [tmp_path / "x.pdf"]) 
    monkeypatch.setattr("src.document_ingestion.data_ingestion.load_documents", lambda paths: [])

    with pytest.raises(Exception):
        ing.built_retriver([io.BytesIO(b"%PDF")], chunk_size=10, chunk_overlap=1, k=3)


def test_chatingestor_happy_path(tmp_path, monkeypatch):
    ing = ChatIngestor(temp_base=str(tmp_path / "data"), faiss_base=str(tmp_path / "faiss"), use_session_dirs=True)

    # Stub file IO and doc loading
    fake_path = tmp_path / "doc.txt"
    fake_path.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr("src.document_ingestion.data_ingestion.save_uploaded_files", lambda uploaded_files, d: [fake_path])
    monkeypatch.setattr("src.document_ingestion.data_ingestion.load_documents", lambda paths: [Document(page_content="text", metadata={})])

    # Stub FAISS behavior via manager methods
    class FakeVS:
        def as_retriever(self, search_type="similarity", search_kwargs=None):
            class R:
                def __init__(self):
                    self.search_kwargs = search_kwargs or {"k": 5}
            return R()
        def save_local(self, *args, **kwargs):
            pass
        def add_documents(self, docs):
            pass

    def fake_load_or_create(*args, **kwargs):
        return FakeVS()

    class FakeFM:
        def __init__(self, *args, **kwargs):
            pass
        load_or_create = staticmethod(fake_load_or_create)
        def add_documents(self, docs):
            return len(docs)

    monkeypatch.setattr("src.document_ingestion.data_ingestion.FaissManager", FakeFM)

    retriever = ing.built_retriver([io.BytesIO(b"data")], chunk_size=50, chunk_overlap=5, k=7)
    assert isinstance(retriever, object)
    assert retriever.search_kwargs.get("k") == 7


def test_dochandler_save_file_and_disallowed_ext(tmp_path):
    dh = DocHandler(data_dir=str(tmp_path))
    # allowed ext
    class F:
        name = "ok.txt"
        def read(self):
            return b"hello"
    p = dh.save_file(F())
    assert Path(p).exists()

    # disallowed ext
    class B:
        name = "bad.exe"
        def read(self):
            return b"x"
    with pytest.raises(Exception):
        dh.save_file(B())


def test_dochandler_read_text_routing(tmp_path, monkeypatch):
    dh = DocHandler(data_dir=str(tmp_path))

    # Create fake files with different extensions
    p_md = Path(dh.session_path) / "a.md"; p_md.write_text("h", encoding="utf-8")
    p_txt = Path(dh.session_path) / "a.txt"; p_txt.write_text("h", encoding="utf-8")
    p_csv = Path(dh.session_path) / "a.csv"; p_csv.write_text("a,b\n1,2", encoding="utf-8")

    # Stub heavy readers
    monkeypatch.setattr(DocHandler, "read_pdf", lambda self, p: "PDF")
    monkeypatch.setattr(DocHandler, "_read_docx", lambda self, p: "DOCX")
    monkeypatch.setattr(DocHandler, "_read_pptx", lambda self, p: "PPTX")
    monkeypatch.setattr(DocHandler, "_read_md", lambda self, p: "MD")
    monkeypatch.setattr(DocHandler, "_read_txt", lambda self, p: "TXT")
    monkeypatch.setattr(DocHandler, "_read_csv", lambda self, p: "CSV")
    monkeypatch.setattr(DocHandler, "_read_xlsx", lambda self, p: "XLSX")
    monkeypatch.setattr(DocHandler, "_read_xls", lambda self, p: "XLS")
    monkeypatch.setattr(DocHandler, "_read_sqlite", lambda self, p: "SQLITE")

    assert dh.read_text(str(Path(dh.session_path) / "x.pdf")) == "PDF"
    assert dh.read_text(str(Path(dh.session_path) / "x.docx")) == "DOCX"
    assert dh.read_text(str(Path(dh.session_path) / "x.pptx")) == "PPTX"
    assert dh.read_text(str(Path(dh.session_path) / "x.md")) == "MD"
    assert dh.read_text(str(Path(dh.session_path) / "x.txt")) == "TXT"
    assert dh.read_text(str(Path(dh.session_path) / "x.csv")) == "CSV"
    assert dh.read_text(str(Path(dh.session_path) / "x.xlsx")) == "XLSX"
    assert dh.read_text(str(Path(dh.session_path) / "x.xls")) == "XLS"
    assert dh.read_text(str(Path(dh.session_path) / "x.sqlite")) == "SQLITE"


def test_dochandler_read_pdf_failure(tmp_path):
    dh = DocHandler(data_dir=str(tmp_path))
    bad = Path(dh.session_path) / "bad.pdf"
    bad.write_bytes(b"not a real pdf")
    with pytest.raises(Exception):
        dh.read_pdf(str(bad))


def test_document_comparator_save_and_combine(tmp_path, monkeypatch):
    dc = DocumentComparator(base_dir=str(tmp_path))

    class F:
        def __init__(self, name, data):
            self.name = name
            self._d = data
        def read(self):
            return self._d

    # Unsupported should raise
    with pytest.raises(Exception):
        dc.save_uploaded_files(F("bad.exe", b"x"), F("ok.pdf", b"x"))

    # Supported save + combine
    ref = F("r.txt", b"hello")
    act = F("a.txt", b"world")
    ref_path, act_path = dc.save_uploaded_files(ref, act)
    assert Path(ref_path).exists() and Path(act_path).exists()

    # Read routes through DocHandler for non-PDF
    from src.document_ingestion.data_ingestion import DocHandler as DH
    monkeypatch.setattr(DH, "read_text", lambda self, p: "CONTENT")
    combined = dc.combine_documents()
    assert "Document: r.txt" in combined and "Document: a.txt" in combined


def test_document_comparator_read_pdf_error(tmp_path):
    dc = DocumentComparator(base_dir=str(tmp_path))
    bad = Path(dc.session_path) / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        dc.read_pdf(bad)


