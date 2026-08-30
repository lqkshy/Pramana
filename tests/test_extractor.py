"""Tests for the claim extractor."""

import pytest

from app.pipeline.claims.extractor import extract_claims


@pytest.mark.asyncio
async def test_extract_claims_returns_correct_keys():
    """extract_claims should return a dict with the expected keys."""
    result = await extract_claims("Elon Musk founded Tesla in 2003")
    assert isinstance(result, dict)
    assert "selected_claims" in result
    assert "disambiguated" in result
    assert "decomposed" in result