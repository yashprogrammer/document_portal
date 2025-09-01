import os
from typing import Optional

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from logger import GLOBAL_LOGGER as log

_INITIALIZED = False


def _enabled() -> bool:
    return os.getenv("LLM_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def init_llm_cache(force: bool = False) -> Optional[str]:
    """
    Initialize a process-wide in-memory cache for LangChain LLM calls.
    Idempotent; respects env toggle LLM_CACHE_ENABLED.

    Returns: 'initialized' | 'already' | 'disabled' | None (on error)
    """
    global _INITIALIZED

    if not _enabled():
        log.info("LLM cache disabled via env", env="LLM_CACHE_ENABLED")
        return "disabled"

    if _INITIALIZED and not force:
        return "already"

    try:
        set_llm_cache(InMemoryCache())
        _INITIALIZED = True
        log.info("LLM in-memory cache initialized")
        return "initialized"
    except Exception as e:
        log.warning("Failed to initialize LLM cache", error=str(e))
        return None


