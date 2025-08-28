## Test Plan: DocumentAnalyzer (P0)

### Scope

- Module: `src/document_analyzer/data_analysis.py`
- Focus: Initialization, primary flow, error wrapping

### Assumptions

- Tests run offline with mocks for LLM/chain.
- `pytest` used; no real network/model calls.

### P0 Test Cases

- test_document_analyzer_init_success
  - Ensures `llm`, `parser`, `fixing_parser`, and `prompt` are initialized.
- test_document_analyzer_init_llm_failure_wrapped
  - When `ModelLoader.load_llm` raises, a `DocumentPortalException` is raised.
- test_analyze_document_happy_path_returns_dict
  - Given a stubbed chain, `.analyze_document(text)` returns a dict matching `Metadata` shape.
- test_analyze_document_passes_expected_inputs_to_chain
  - Verifies `chain.invoke` receives `{"format_instructions": ..., "document_text": <exact_input>}`.
- test_analyze_document_chain_error_wrapped
  - If `chain.invoke` raises, a `DocumentPortalException` is raised.

### Mocking Guidance

- Stub `ModelLoader.load_llm` to a lightweight fake.
- Make `self.prompt | self.llm | self.fixing_parser` yield a stub chain with `invoke(payload)`.
