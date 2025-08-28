# tests/test_unit_cases.py

import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint
from model.models import Metadata
from exception.custom_exception import DocumentPortalException
from src.document_analyzer.data_analysis import DocumentAnalyzer
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text

def test_document_analyzer_init_success(monkeypatch):
    fake_llm = object()

    class FakeLoader:
        def __init__(self): pass
        def load_llm(self):
            return fake_llm

    monkeypatch.setattr("src.document_analyzer.data_analysis.ModelLoader", FakeLoader)
    monkeypatch.setattr(
        "src.document_analyzer.data_analysis.OutputFixingParser.from_llm",
        lambda parser, llm: "FIX_PARSER",
        raising=True,
    )

    analyzer = DocumentAnalyzer()

    assert analyzer.llm is fake_llm
    assert isinstance(analyzer.parser, JsonOutputParser)
    assert analyzer.fixing_parser == "FIX_PARSER"
    assert isinstance(analyzer.prompt, ChatPromptTemplate)


def test_document_analyzer_init_llm_failure_wrapped(monkeypatch):
    # Make the internal ModelLoader return a loader whose load_llm() fails
    class FakeLoader:
        def __init__(self): pass
        def load_llm(self):
            raise RuntimeError("LLM init failed")

    monkeypatch.setattr("src.document_analyzer.data_analysis.ModelLoader", FakeLoader)

    from src.document_analyzer.data_analysis import DocumentAnalyzer

    with pytest.raises(DocumentPortalException) as exc:
        DocumentAnalyzer()

    assert "Error in DocumentAnalyzer initialization" in str(exc.value)


def test_analyze_document_happy_path_returns_dict(monkeypatch):
    # Stub the internal ModelLoader to avoid real LLM loading
    class FakeLoader:
        def __init__(self): pass
        def load_llm(self):
            return object()

    monkeypatch.setattr("src.document_analyzer.data_analysis.ModelLoader", FakeLoader)

    # Avoid constructing a real fixing parser
    monkeypatch.setattr(
        "src.document_analyzer.data_analysis.OutputFixingParser.from_llm",
        lambda parser, llm: "FIX_PARSER",
        raising=True,
    )

    # Fake prompt and chain to control .invoke output
    class FakeChain:
        def __or__(self, other):
            return self
        def invoke(self, payload):
            return {
                "Summary": ["Concise summary."],
                "Title": "Sample Doc",
                "Author": ["Alice", "Bob"],
                "DateCreated": "2024-01-01",
                "LastModifiedDate": "2024-02-01",
                "Publisher": "ACME",
                "Language": "en",
                "PageCount": 10,
                "SentimentTone": "neutral",
            }

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    from src.document_analyzer.data_analysis import DocumentAnalyzer
    analyzer = DocumentAnalyzer()
    analyzer.prompt = FakePrompt()  # inject our fake chain source

    result = analyzer.analyze_document("dummy text")

    assert isinstance(result, dict)
    # Validate shape by parsing with the Pydantic model
    parsed = Metadata(**result)
    assert parsed.Title == "Sample Doc"
    assert isinstance(parsed.Summary, list)