"""
tests/test_claim_matching.py
──────────────────────────────
Tests for Stage 3: Claim matching (MiniLM + in-memory fallback).

These tests use the in-memory fallback store (no DATABASE_URL needed).
Run with:  pytest tests/test_claim_matching.py -v

Requires: pip install sentence-transformers
"""

import pytest

from app.models.schemas import VerdictLabel, VerificationResult
from app.pipeline.claim_matching import (
    clear_memory_store,
    get_memory_store_size,
    match_claim,
    store_claim,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
def _make_result(claim: str, verdict: VerdictLabel = VerdictLabel.SUPPORTED) -> VerificationResult:
    return VerificationResult(
        claim=claim,
        verdict=verdict,
        confidence=0.95,
        explanation="Test explanation.",
        evidence_passages=[],
        cited_source_urls=[],
        evidence_strength=0.80,
    )


@pytest.fixture(autouse=True)
def reset_memory_store():
    """Clear in-memory store before every test to prevent cross-test contamination."""
    clear_memory_store()
    yield
    clear_memory_store()


# ── Tests ──────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_new_claim_returns_no_match():
    """A claim with nothing in the store should return matched=False."""
    result = await match_claim("The Eiffel Tower is located in Paris, France.")
    assert result.matched is False
    assert result.cached_verdict is None


@pytest.mark.asyncio
async def test_identical_claim_returns_cache_hit():
    """
    A claim stored and then queried with an identical string should return matched=True.
    Identical text → cosine similarity ≈ 1.0 (above 0.92 threshold).
    """
    claim = "NASA was founded on July 29, 1958."
    result_obj = _make_result(claim)
    await store_claim("test-id-001", claim, result_obj)

    assert get_memory_store_size() == 1

    match = await match_claim(claim)
    assert match.matched is True
    assert match.cached_verdict is not None
    assert match.similarity_score > 0.92


@pytest.mark.asyncio
async def test_paraphrase_returns_cache_hit():
    """
    A near-paraphrase of a stored claim should be a cache hit (similarity > 0.92).
    MiniLM handles semantic similarity, not just lexical overlap.
    """
    original = "Apple's market capitalisation exceeded two trillion dollars in 2020."
    paraphrase = "Apple's market cap surpassed $2 trillion in the year 2020."

    result_obj = _make_result(original)
    await store_claim("test-id-002", original, result_obj)

    match = await match_claim(paraphrase)
    # MiniLM should recognise these as semantically equivalent.
    assert match.matched is True
    assert match.similarity_score > 0.92


@pytest.mark.asyncio
async def test_different_claim_returns_miss():
    """
    An unrelated claim should not match even with something in the store.
    """
    stored_claim = "The Great Wall of China is over 13,000 miles long."
    result_obj = _make_result(stored_claim)
    await store_claim("test-id-003", stored_claim, result_obj)

    different_claim = "The unemployment rate in Germany fell to 5.1% in Q3 2023."
    match = await match_claim(different_claim)
    assert match.matched is False


@pytest.mark.asyncio
async def test_store_multiple_claims_picks_closest():
    """
    With multiple claims in the store, matching should return the most similar one.
    """
    claims = [
        ("test-001", "Tesla delivered 1.8 million cars in 2023."),
        ("test-002", "The population of China is approximately 1.4 billion."),
        ("test-003", "Bitcoin reached an all-time high of $69,000 in November 2021."),
    ]
    for claim_id, claim in claims:
        await store_claim(claim_id, claim, _make_result(claim))

    # Query semantically similar to the Tesla claim.
    query = "Tesla sold 1.8 million vehicles globally in fiscal year 2023."
    match = await match_claim(query)
    assert match.matched is True
    assert match.cached_verdict is not None
    assert "Tesla" in match.cached_verdict.claim or "1.8 million" in match.cached_verdict.claim


@pytest.mark.asyncio
async def test_store_increments_memory_size():
    """store_claim should increment the in-memory store count."""
    assert get_memory_store_size() == 0

    claim = "The moon is approximately 384,400 km from Earth."
    await store_claim("test-id-004", claim, _make_result(claim))
    assert get_memory_store_size() == 1

    claim2 = "Mars is the fourth planet from the Sun."
    await store_claim("test-id-005", claim2, _make_result(claim2))
    assert get_memory_store_size() == 2


@pytest.mark.asyncio
async def test_cached_verdict_carries_correct_fields():
    """The cached_verdict returned on a hit should have the correct fields."""
    claim = "The speed of light in a vacuum is approximately 299,792 km/s."
    original_result = _make_result(claim, verdict=VerdictLabel.SUPPORTED)
    original_result.confidence = 0.99
    original_result.evidence_strength = 0.85

    await store_claim("test-id-006", claim, original_result)
    match = await match_claim(claim)

    assert match.matched is True
    cached = match.cached_verdict
    assert cached is not None
    assert cached.verdict == VerdictLabel.SUPPORTED
    assert cached.confidence == pytest.approx(0.99)
    assert cached.evidence_strength == pytest.approx(0.85)
