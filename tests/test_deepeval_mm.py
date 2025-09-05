import pytest


@pytest.mark.deepeval
def test_mm_deepeval_runner_imports():
    # Sanity check to ensure the runner can be imported; actual evaluation is run in CI
    import importlib
    mod = importlib.import_module("eval.run_mm_doc_chat_deepeval")
    assert hasattr(mod, "main")


