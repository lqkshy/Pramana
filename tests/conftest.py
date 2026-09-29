"""Shared pytest fixtures for the Pramana test suite."""

import importlib.util
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Remember whether the developer has a real Groq key BEFORE we add a dummy one.
_HAS_REAL_GROQ_KEY = bool(os.getenv("GROQ_API_KEY"))
os.environ.setdefault("GROQ_API_KEY", "gsk-test-dummy-key-for-testing-only")

_NEEDS_GROQ_MODULES = {"test_check_worthiness"}
_NEEDS_GROQ_PREFIXES = ("test_generate_queries",)          # integration tests in test_queries.py
_NEEDS_EMBEDDER_MODULES = {"test_claim_matching"}


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that need real API keys / heavy models that aren't available."""
    skip_groq = pytest.mark.skip(reason="GROQ_API_KEY not set in .env — skipping live LLM tests.")
    skip_embed = pytest.mark.skip(reason="sentence-transformers not installed.")
    has_embedder = importlib.util.find_spec("sentence_transformers") is not None

    for item in items:
        module = item.module.__name__.split(".")[-1]
        if not _HAS_REAL_GROQ_KEY and (
            module in _NEEDS_GROQ_MODULES
            or item.name.startswith(_NEEDS_GROQ_PREFIXES)
        ):
            item.add_marker(skip_groq)
        if not has_embedder and module in _NEEDS_EMBEDDER_MODULES:
            item.add_marker(skip_embed)
