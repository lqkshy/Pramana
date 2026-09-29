"""
tests/test_check_worthiness.py
────────────────────────────────
Tests for Stage 2: Check-worthiness classification.

Run with:  pytest tests/test_check_worthiness.py -v

These tests make real Groq API calls.
Set GROQ_API_KEY in .env before running.
DEV_MODE=true by default — uses Llama 3.1 8B (safe, 14,400 req/day).
"""

import asyncio

import pytest

from app.pipeline.check_worthiness import check_worthiness, filter_check_worthy


# ── Check-worthy claims (should return True) ───────────────────────────────────
CHECK_WORTHY_CASES = [
    "The unemployment rate in the United States fell to 3.7% in September 2023.",
    "Apple's revenue exceeded $383 billion in fiscal year 2023.",
    "The COVID-19 pandemic began in Wuhan, China in late 2019.",
    "Mount Everest is 8,848.86 metres tall.",
    "NASA's Artemis I mission launched on November 16, 2022.",
    "The World Health Organization declared COVID-19 a pandemic on March 11, 2020.",
    "Tesla delivered 1.8 million vehicles in 2023.",
]

# ── Not check-worthy (should return False) ─────────────────────────────────────
NOT_WORTHY_CASES = [
    "Climate change is the most important issue of our time.",      # Opinion
    "AI will probably replace most jobs in the next decade.",       # Prediction
    "Isn't it obvious that politicians never tell the truth?",      # Rhetorical question
    "Things are generally getting better.",                         # Vague generality
    "Democracy is the best form of government.",                    # Normative statement
    "Prices are high.",                                             # Too vague
]


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", CHECK_WORTHY_CASES)
async def test_check_worthy_claims_pass(claim: str):
    """Specific, falsifiable, real-world claims should be marked check-worthy."""
    result = await check_worthiness(claim)
    assert result.is_check_worthy is True, (
        f"Expected check-worthy=True for claim: {claim!r}\n"
        f"Got reason: {result.reason}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", NOT_WORTHY_CASES)
async def test_non_worthy_claims_filtered(claim: str):
    """Opinions, predictions, and vague claims should be filtered out."""
    result = await check_worthiness(claim)
    assert result.is_check_worthy is False, (
        f"Expected check-worthy=False for claim: {claim!r}\n"
        f"Got reason: {result.reason}"
    )


@pytest.mark.asyncio
async def test_result_has_reason():
    """Every result should include a non-empty reason string."""
    claim = "The Eiffel Tower is located in Paris, France."
    result = await check_worthiness(claim)
    assert isinstance(result.reason, str)
    assert len(result.reason) > 5, "Reason should be a non-trivial string."


@pytest.mark.asyncio
async def test_filter_check_worthy_removes_opinions():
    """filter_check_worthy should remove non-worthy claims from a mixed list."""
    mixed = [
        "The Eiffel Tower is 330 metres tall.",              # worthy
        "Paris is the most romantic city in the world.",      # not worthy — opinion
        "NASA was founded on July 29, 1958.",                 # worthy
    ]
    worthy = await filter_check_worthy(mixed)
    assert len(worthy) == 2
    assert any("Eiffel" in c for c in worthy)
    assert any("NASA" in c for c in worthy)
    assert not any("romantic" in c for c in worthy)


@pytest.mark.asyncio
async def test_empty_result_for_all_opinions():
    """All opinions should produce an empty list."""
    all_opinions = [
        "Technology is wonderful.",
        "People should be kinder.",
        "The future will be amazing.",
    ]
    worthy = await filter_check_worthy(all_opinions)
    assert worthy == []
