"""Tests for the LLM client (no network — providers are mocked)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services import llm_client


@pytest.mark.asyncio
async def test_call_llm_returns_string():
    with patch("app.services.llm_client._call_groq", new=AsyncMock(return_value="hello")):
        result = await llm_client.call_llm("Say hello in one word", task_type="fast")
    assert isinstance(result, str) and result == "hello"


@pytest.mark.asyncio
async def test_call_llm_forwards_all_pipeline_kwargs():
    """Every pipeline stage passes these kwargs — they must reach the provider."""
    mock = AsyncMock(return_value="{}")
    with patch("app.services.llm_client._call_groq", new=mock):
        await llm_client.call_llm(
            prompt="p", task_type="fast", system_prompt="sys",
            max_tokens=77, temperature=0.2, json_mode=True,
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["system_prompt"] == "sys"
    assert kwargs["max_tokens"] == 77
    assert kwargs["temperature"] == 0.2
    assert kwargs["json_mode"] is True


@pytest.mark.asyncio
async def test_reasoning_routes_by_dev_mode(monkeypatch):
    mock = AsyncMock(return_value="ok")
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "groq")
    with patch("app.services.llm_client._call_groq", new=mock):
        monkeypatch.setattr(llm_client, "DEV_MODE", True)
        await llm_client.call_llm("p", task_type="reasoning")
        assert mock.call_args.args[1] == "llama-3.1-8b-instant"
        monkeypatch.setattr(llm_client, "DEV_MODE", False)
        await llm_client.call_llm("p", task_type="reasoning")
        assert mock.call_args.args[1] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_unknown_task_type_raises():
    with pytest.raises(ValueError):
        await llm_client.call_llm("p", task_type="banana")


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(monkeypatch):
    async def instant_sleep(_):
        return None

    monkeypatch.setattr(llm_client.asyncio, "sleep", instant_sleep)
    mock = AsyncMock(side_effect=[Exception("Error 429 rate limit, try again in 1s"), "recovered"])
    with patch("app.services.llm_client._call_groq", new=mock):
        assert await llm_client.call_llm("p") == "recovered"
    assert mock.call_count == 2
