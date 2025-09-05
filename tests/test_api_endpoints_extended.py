import io
import os
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "document-portal"}


def test_protected_unauthorized():
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_protected_authorized(monkeypatch):
    class FakeUser:
        def __init__(self):
            self.id = "user-123"

    # Override the dependency used in api.main
    from api import main as api_main

    app.dependency_overrides[api_main.current_active_user] = lambda: FakeUser()
    try:
        resp = client.get("/protected")
        assert resp.status_code == 200
        js = resp.json()
        assert js.get("user_id") == "user-123"
    finally:
        app.dependency_overrides.pop(api_main.current_active_user, None)


def test_pages_and_redirects():
    # Public pages
    for path in ["/", "/login", "/signup"]:
        resp = client.get(path)
        assert resp.status_code == 200
        assert isinstance(resp.text, str) and len(resp.text) > 0

    # /app should redirect to /login when unauthenticated
    resp = client.get("/app", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers.get("location") == "/login"


def test_analyze_unauthorized(monkeypatch):
    # Force _is_test_request to False so auth is enforced in tests
    monkeypatch.setattr("api.main._is_test_request", lambda request: False)

    fake_pdf = io.BytesIO(b"%PDF-1.4 test")
    files = {"file": ("sample.pdf", fake_pdf, "application/pdf")}
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 401


def test_analyze_happy_path(monkeypatch):
    class FakeDocHandler:
        def __init__(self, *args, **kwargs):
            pass
        def save_file(self, uploaded_file):
            return "/tmp/fake.pdf"
        def read_text(self, path: str):
            return "some text"

    class FakeAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze_document(self, text):
            return {
                "Title": "Doc",
                "Summary": ["S"],
                "Author": ["A"],
                "DateCreated": "2024-01-01",
                "LastModifiedDate": "2024-01-02",
                "Publisher": "P",
                "Language": "en",
                "PageCount": 1,
                "SentimentTone": "neutral",
            }

    monkeypatch.setattr("api.main.DocHandler", FakeDocHandler)
    monkeypatch.setattr("api.main.DocumentAnalyzer", FakeAnalyzer)

    fake_pdf = io.BytesIO(b"%PDF-1.4 test")
    files = {"file": ("sample.pdf", fake_pdf, "application/pdf")}
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("Title") == "Doc"
    assert isinstance(js.get("Summary"), list)


def test_analyze_failure(monkeypatch):
    class FakeDocHandler:
        def __init__(self, *args, **kwargs):
            pass
        def save_file(self, uploaded_file):
            return "/tmp/fake.pdf"
        def read_text(self, path: str):
            return "some text"

    class FailAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze_document(self, text):
            raise RuntimeError("boom")

    monkeypatch.setattr("api.main.DocHandler", FakeDocHandler)
    monkeypatch.setattr("api.main.DocumentAnalyzer", FailAnalyzer)

    fake_pdf = io.BytesIO(b"%PDF-1.4 test")
    files = {"file": ("sample.pdf", fake_pdf, "application/pdf")}
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 500
    assert "Analysis failed" in resp.json().get("detail", "")


def test_compare_unauthorized(monkeypatch):
    monkeypatch.setattr("api.main._is_test_request", lambda request: False)

    ref = io.BytesIO(b"a")
    act = io.BytesIO(b"b")
    files = {
        "reference": ("ref.pdf", ref, "application/pdf"),
        "actual": ("act.pdf", act, "application/pdf"),
    }
    resp = client.post("/compare", files=files)
    assert resp.status_code == 401


def test_compare_happy_path(monkeypatch):
    class FakeComparator:
        def __init__(self, *args, **kwargs):
            self.session_id = "sess-1"
        def save_uploaded_files(self, ref, act):
            return ("/tmp/ref.pdf", "/tmp/act.pdf")
        def combine_documents(self):
            return "combined"

    class FakeDf:
        def to_dict(self, orient="records"):
            return [{"a": 1}]

    class FakeLLMComparator:
        def __init__(self, *args, **kwargs):
            pass
        def compare_documents(self, text):
            return FakeDf()

    monkeypatch.setattr("api.main.DocumentComparator", FakeComparator)
    monkeypatch.setattr("api.main.DocumentComparatorLLM", FakeLLMComparator)

    ref = io.BytesIO(b"a")
    act = io.BytesIO(b"b")
    files = {
        "reference": ("ref.pdf", ref, "application/pdf"),
        "actual": ("act.pdf", act, "application/pdf"),
    }
    resp = client.post("/compare", files=files)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("session_id") == "sess-1"
    assert js.get("rows") == [{"a": 1}]


def test_compare_failure_invalid_types():
    ref = io.BytesIO(b"a")
    act = io.BytesIO(b"b")
    files = {
        "reference": ("ref.exe", ref, "application/octet-stream"),
        "actual": ("act.exe", act, "application/octet-stream"),
    }
    resp = client.post("/compare", files=files)
    assert resp.status_code == 500
    assert "Comparison failed" in resp.json().get("detail", "")


def test_chat_index_non_mm_happy_path(monkeypatch):
    class FakeRetriever:
        def __init__(self):
            self.search_kwargs = {"k": 5}

    class FakeChatIngestor:
        def __init__(self, *args, **kwargs):
            self.session_id = "s1"
        def built_retriver(self, *args, **kwargs):
            return FakeRetriever()

    monkeypatch.setattr("api.main.ChatIngestor", FakeChatIngestor)

    fake_pdf = io.BytesIO(b"%PDF-1.4 test")
    files = {"files": ("sample.pdf", fake_pdf, "application/pdf")}
    data = {"multimodal": "false", "use_session_dirs": "true", "k": "4", "chunk_size": "1000", "chunk_overlap": "200"}
    resp = client.post("/chat/index", files=files, data=data)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("multimodal") is False
    assert js.get("session_id") == "s1"


def test_chat_index_failure(monkeypatch):
    class FakeChatIngestor:
        def __init__(self, *args, **kwargs):
            self.session_id = "s1"
        def built_retriver(self, *args, **kwargs):
            raise RuntimeError("index failure")

    monkeypatch.setattr("api.main.ChatIngestor", FakeChatIngestor)

    fake_pdf = io.BytesIO(b"%PDF-1.4 test")
    files = {"files": ("sample.pdf", fake_pdf, "application/pdf")}
    data = {"multimodal": "false", "use_session_dirs": "true"}
    resp = client.post("/chat/index", files=files, data=data)
    assert resp.status_code == 500
    assert "Indexing failed" in resp.json().get("detail", "")


def test_chat_query_missing_session_id():
    data = {"question": "hi", "use_session_dirs": "true", "multimodal": "false"}
    resp = client.post("/chat/query", data=data)
    assert resp.status_code == 400


def test_chat_query_non_mm_missing_index(tmp_path):
    # Ensure directory does not exist
    session_id = "nonexistent-session"
    data = {"question": "q", "use_session_dirs": "true", "multimodal": "false", "session_id": session_id}
    resp = client.post("/chat/query", data=data)
    assert resp.status_code == 404


def test_chat_query_non_mm_happy_path(monkeypatch, tmp_path):
    # Create expected index dir to bypass 404 check
    session_id = "sess-ok"
    base = "faiss_index"
    os.makedirs(os.path.join(base, session_id), exist_ok=True)

    class FakeRAG:
        def __init__(self, session_id=None, retriever=None):
            self.session_id = session_id
        def load_retriever_from_faiss(self, *args, **kwargs):
            return object()
        def invoke(self, question, chat_history=None):
            return "answer text"

    monkeypatch.setattr("api.main.ConversationalRAG", FakeRAG)

    data = {
        "question": "hi",
        "use_session_dirs": "true",
        "multimodal": "false",
        "session_id": session_id,
        "k": "3",
    }
    resp = client.post("/chat/query", data=data)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("engine") == "LCEL-RAG"
    assert js.get("answer") == "answer text"


def test_chat_query_mm_failure(monkeypatch, tmp_path):
    # Provide fake retriever but make chain building fail
    class FakeRetriever:
        def __init__(self):
            self.search_kwargs = {"k": 5}

    def fake_load_retriever(path, loader):
        return FakeRetriever()

    def boom_build_chain(*args, **kwargs):
        raise RuntimeError("chain error")

    monkeypatch.setattr("api.main.load_multimodal_retriever", fake_load_retriever)
    monkeypatch.setattr("api.main.build_multimodal_chain", boom_build_chain)

    data = {"multimodal": "true", "question": "hello", "use_session_dirs": "false"}
    resp = client.post("/chat/query", data=data)
    assert resp.status_code == 500
    assert "Query failed" in resp.json().get("detail", "")


