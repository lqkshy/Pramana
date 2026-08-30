"""Tests for the LLM client."""

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_call_llm_returns_string():
    """call_llm should return a non-empty string for a simple prompt."""
    with patch("app.services.llm_client._call_groq", return_value="hello"):
        from app.services.llm_client import call_llm

        result = await call_llm("Say hello in one word", task_type="fast")
    assert isinstance(result, str)
    assert len(result) > 0