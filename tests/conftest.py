"""Shared pytest fixtures for the Pramana test suite."""

import os
import pytest

# Ensure .env is loaded and API keys are available for tests
os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
from dotenv import load_dotenv
load_dotenv()

# Set a dummy GROQ_API_KEY so LLM calls don't fail in tests
os.environ.setdefault("GROQ_API_KEY", "sk-test-dummy-key-for-testing-only")


@pytest.fixture
def test_client():
    """Return a FastAPI TestClient wrapping the Pramana app."""
    from main import app
    return TestClient(app)