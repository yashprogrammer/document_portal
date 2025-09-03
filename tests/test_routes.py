import io
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_index_build_multimodal_flag(monkeypatch):
    # Prepare a tiny PDF-like file object (the server won't parse it deeply in this test)
    fake_pdf = io.BytesIO(b"%PDF-1.4 test")
    files = {"files": ("sample.pdf", fake_pdf, "application/pdf")}

    # Monkeypatch the ingestor to avoid heavy processing
    class FakeMMIngestor:
        def __init__(self, *args, **kwargs):
            self.session_id = "test-session"
        def built_retriver(self, *args, **kwargs):
            return None

    monkeypatch.setattr("api.main.MultiModalChatIngestor", FakeMMIngestor)

    data = {
        "multimodal": "true",
        "use_session_dirs": "true",
        "k": "3",
    }
    resp = client.post("/chat/index", files=files, data=data)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("multimodal") is True
    assert js.get("session_id") == "test-session"


def test_query_multimodal_flag(monkeypatch):
    # Fake loader pieces to bypass real LLM and stores
    class FakeLoader:
        def __init__(self): pass
        def load_llm(self):
            class L:
                def __or__(self, other): return self
                def invoke(self, q): return "ok"
            return L()

    class FakeRetriever:
        def __or__(self, other): return self

    def fake_load_retriever(path, loader):
        return FakeRetriever()

    def fake_build_chain(ret, llm):
        class C:
            def invoke(self, q): return "answer"
        return C()

    monkeypatch.setattr("api.main.ModelLoader", FakeLoader)
    monkeypatch.setattr("api.main.load_multimodal_retriever", fake_load_retriever)
    monkeypatch.setattr("api.main.build_multimodal_chain", fake_build_chain)

    data = {
        "multimodal": "true",
        "question": "hello",
        "use_session_dirs": "false",
    }
    resp = client.post("/chat/query", data=data)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("engine") == "MM-LCEL-RAG"
    assert js.get("answer") == "answer"


