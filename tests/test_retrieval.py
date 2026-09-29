"""
tests/test_retrieval.py
────────────────────────
Tests for Stage 5: Retrieval (dual-mode Tavily / pre-retrieved).

Run with:  pytest tests/test_retrieval.py -v

The pre-retrieved tests run without any API keys.
The live Tavily tests require TAVILY_API_KEY in .env.
"""

import os

import pytest

from app.pipeline.retrieval import retrieve_evidence, set_live_search
from app.models.schemas import RetrievalResult


# ── Pre-retrieved mode tests (no API keys needed) ─────────────────────────────
class TestPreRetrievedMode:
    """All tests in this class use USE_LIVE_SEARCH=false."""

    @pytest.fixture(autouse=True)
    def force_eval_mode(self):
        """Force pre-retrieved mode for every test in this class."""
        original = os.getenv("USE_LIVE_SEARCH", "false")
        set_live_search(False)
        yield
        os.environ["USE_LIVE_SEARCH"] = original

    @pytest.mark.asyncio
    async def test_pre_retrieved_returns_correct_mode(self):
        """Result mode should be 'pre_retrieved' when USE_LIVE_SEARCH=false."""
        evidence = [
            {"url": "https://example.com/article1", "snippet": "Test snippet 1."},
            {"url": "https://example.com/article2", "snippet": "Test snippet 2."},
        ]
        result = await retrieve_evidence(
            claim="Test claim",
            queries=["test query"],
            pre_retrieved_evidence=evidence,
        )
        assert result.mode == "pre_retrieved"

    @pytest.mark.asyncio
    async def test_pre_retrieved_passes_through_all_sources(self):
        """All supplied evidence items should appear in the result."""
        evidence = [
            {"url": "https://reuters.com/1", "snippet": "Reuters snippet."},
            {"url": "https://bbc.com/2", "snippet": "BBC snippet."},
            {"url": "https://nytimes.com/3", "snippet": "NYT snippet."},
        ]
        result = await retrieve_evidence(
            claim="WHO declared COVID-19 a pandemic.",
            queries=["WHO COVID pandemic 2020"],
            pre_retrieved_evidence=evidence,
        )
        assert len(result.sources) == 3
        urls = {s.url for s in result.sources}
        assert "https://reuters.com/1" in urls
        assert "https://bbc.com/2" in urls

    @pytest.mark.asyncio
    async def test_pre_retrieved_handles_empty_evidence(self):
        """Empty pre-retrieved list should return empty sources, not an error."""
        result = await retrieve_evidence(
            claim="Some claim",
            queries=["some query"],
            pre_retrieved_evidence=[],
        )
        assert result.mode == "pre_retrieved"
        assert len(result.sources) == 0

    @pytest.mark.asyncio
    async def test_pre_retrieved_no_queries_required(self):
        """Pre-retrieved mode should work even if queries list is empty."""
        evidence = [{"url": "https://example.com", "snippet": "Evidence here."}]
        result = await retrieve_evidence(
            claim="Some claim",
            queries=[],           # No queries needed in eval mode
            pre_retrieved_evidence=evidence,
        )
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_pre_retrieved_populates_snippet_from_text_key(self):
        """Items with 'text' key (SciFact format) should map to snippet."""
        evidence = [
            {"text": "SciFact abstract text here."},   # No URL, no 'snippet' key
        ]
        result = await retrieve_evidence(
            claim="Test claim",
            queries=[],
            pre_retrieved_evidence=evidence,
        )
        assert len(result.sources) == 1
        assert result.sources[0].snippet == "SciFact abstract text here."
        assert result.sources[0].url.startswith("pre_retrieved://")

    @pytest.mark.asyncio
    async def test_pre_retrieved_does_not_call_tavily(self, monkeypatch):
        """
        Verify Tavily is never called in pre-retrieved mode.
        Monkeypatching _tavily_request ensures this test never requires an API key.
        """
        call_count = {"n": 0}

        async def mock_tavily(query):
            call_count["n"] += 1
            return []

        monkeypatch.setattr("app.pipeline.retrieval._tavily_request", mock_tavily)

        await retrieve_evidence(
            claim="Test claim",
            queries=["ignored query"],
            pre_retrieved_evidence=[{"url": "https://example.com", "snippet": "test"}],
        )
        assert call_count["n"] == 0, "Tavily should not be called in pre-retrieved mode."


# ── Live Tavily mode tests (require TAVILY_API_KEY) ────────────────────────────
@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set — skipping live Tavily tests.",
)
class TestLiveSearchMode:
    """These tests make real Tavily API calls. Use sparingly to conserve quota."""

    @pytest.fixture(autouse=True)
    def force_live_mode(self):
        original = os.getenv("USE_LIVE_SEARCH", "false")
        set_live_search(True)
        yield
        os.environ["USE_LIVE_SEARCH"] = original

    @pytest.mark.asyncio
    async def test_live_search_returns_sources(self):
        """Live Tavily search should return at least one source."""
        result = await retrieve_evidence(
            claim="NASA's Artemis I mission launched in November 2022.",
            queries=["NASA Artemis I launch 2022", "Artemis mission moon NASA"],
        )
        assert result.mode == "live"
        assert len(result.sources) >= 1

    @pytest.mark.asyncio
    async def test_live_search_deduplicates_urls(self):
        """Duplicate URLs from multiple queries should be deduplicated."""
        # Same query twice → Tavily may return the same URLs.
        result = await retrieve_evidence(
            claim="Apple revenue 2023",
            queries=["Apple revenue fiscal year 2023", "Apple annual revenue 2023"],
        )
        urls = [s.url for s in result.sources]
        assert len(urls) == len(set(urls)), "No duplicate URLs should appear."

    @pytest.mark.asyncio
    async def test_live_search_sources_have_required_fields(self):
        """Each returned source should have a non-empty URL."""
        result = await retrieve_evidence(
            claim="The population of China exceeded 1.4 billion.",
            queries=["China population 2023 statistics"],
        )
        for source in result.sources:
            assert source.url.startswith("http"), f"Invalid URL: {source.url}"
