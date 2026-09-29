"""
tests/test_queries.py
──────────────────────
Tests for Stage 4: Query generation.

Run with:  pytest tests/test_queries.py -v

Makes real Groq API calls.
Requires GROQ_API_KEY in .env.
"""

import pytest

from app.pipeline.queries import _keyword_fallback, _parse_queries, generate_queries


# ── Unit tests (no API calls) ──────────────────────────────────────────────────
class TestParseQueries:
    def test_valid_json_two_queries(self):
        raw = '{"queries": ["NASA moon landing 1969", "Apollo 11 mission facts"]}'
        result = _parse_queries(raw, "Apollo 11 landed on the moon in 1969.")
        assert len(result) == 2
        assert "NASA" in result[0] or "Apollo" in result[0]

    def test_valid_json_three_queries(self):
        raw = '{"queries": ["q1", "q2", "q3"]}'
        result = _parse_queries(raw, "test claim")
        assert len(result) == 3

    def test_caps_at_three_queries(self):
        raw = '{"queries": ["q1", "q2", "q3", "q4", "q5"]}'
        result = _parse_queries(raw, "test claim")
        assert len(result) == 3  # Capped at 3

    def test_markdown_fence_stripped(self):
        raw = '```json\n{"queries": ["query one", "query two"]}\n```'
        result = _parse_queries(raw, "test claim")
        assert len(result) == 2

    def test_malformed_json_falls_back_to_quoted_strings(self):
        raw = 'I suggest: "Tesla stock price history" and "Tesla 2023 annual report revenue"'
        result = _parse_queries(raw, "Tesla revenue in 2023")
        assert len(result) >= 2

    def test_completely_unparseable_uses_keyword_fallback(self):
        raw = "Here are some ideas for your search"  # No JSON, no quoted strings
        result = _parse_queries(raw, "The population of India exceeded 1.4 billion in 2023")
        assert len(result) == 2  # keyword_fallback always returns exactly 2

    def test_empty_queries_filtered(self):
        raw = '{"queries": ["valid query", "   ", "another valid query"]}'
        result = _parse_queries(raw, "test claim")
        assert all(q.strip() for q in result)


class TestKeywordFallback:
    def test_long_claim_returns_two_different_queries(self):
        claim = "The unemployment rate in the United States fell to 3.7% in September 2023."
        result = _keyword_fallback(claim)
        assert len(result) == 2
        assert result[0] != result[1]

    def test_short_claim_still_returns_two_queries(self):
        claim = "Paris is in France."
        result = _keyword_fallback(claim)
        assert len(result) == 2

    def test_short_claim_deduplicates(self):
        claim = "Yes."
        result = _keyword_fallback(claim)
        assert len(result) == 2  # Deduplication adds " evidence source" to second


# ── Integration tests (real API calls) ────────────────────────────────────────
@pytest.mark.asyncio
async def test_generate_queries_returns_two_to_three():
    """generate_queries must return 2 or 3 queries."""
    claim = "The Great Barrier Reef covers approximately 344,400 square kilometres."
    result = await generate_queries(claim)
    assert 2 <= len(result.queries) <= 3


@pytest.mark.asyncio
async def test_generate_queries_are_strings():
    claim = "Tesla's revenue in Q4 2023 was $25.2 billion."
    result = await generate_queries(claim)
    for q in result.queries:
        assert isinstance(q, str)
        assert len(q.strip()) > 5


@pytest.mark.asyncio
async def test_generate_queries_not_identical():
    """Queries should be diverse — not all identical strings."""
    claim = "Apple became the first company to reach a market cap of $3 trillion."
    result = await generate_queries(claim)
    assert len(set(result.queries)) == len(result.queries), "All queries should be unique."


@pytest.mark.asyncio
async def test_generate_queries_avoids_fact_check_phrases():
    """Queries should not contain 'fact check' or 'is it true' as instructed."""
    claim = "The WHO declared COVID-19 a pandemic on March 11, 2020."
    result = await generate_queries(claim)
    for q in result.queries:
        assert "fact check" not in q.lower()
        assert "is it true" not in q.lower()


@pytest.mark.asyncio
async def test_generate_queries_includes_claim_entities():
    """At least one query should include key terms from the claim."""
    claim = "SpaceX launched the Falcon 9 rocket in February 2024."
    result = await generate_queries(claim)
    all_queries_text = " ".join(result.queries).lower()
    # At least one of the key entities should appear
    assert any(
        term in all_queries_text
        for term in ["spacex", "falcon", "rocket", "2024"]
    ), f"Expected key entities in queries. Got: {result.queries}"


@pytest.mark.asyncio
async def test_generate_queries_batch():
    """Batch generation should return one result per claim."""
    from app.pipeline.queries import generate_queries_batch

    claims = [
        "The moon is 384,400 km from Earth.",
        "Python was created by Guido van Rossum in 1991.",
    ]
    results = await generate_queries_batch(claims)
    assert len(results) == 2
    for r in results:
        assert 2 <= len(r.queries) <= 3
