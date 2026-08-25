"""Tests for extract_claims pipeline function."""

from app.pipeline.claims import extract_claims


def test_valid_input():
    """Call extract_claims with a paragraph and assert result is a non-empty list of strings."""
    claims = extract_claims("GPT-4 is 40% cheaper than Claude 3 Opus and runs at 2x the speed.")
    assert isinstance(claims, list)
    assert len(claims) > 0
    for claim in claims:
        assert isinstance(claim, str)


def test_empty_input():
    """Call extract_claims with empty string and assert it raises ValueError."""
    try:
        extract_claims("")
        assert False, "Expected ValueError to be raised"
    except ValueError:
        pass


def test_returns_list():
    """Call extract_claims with any sentence and assert the return type is list."""
    result = extract_claims("Test claim here.")
    assert isinstance(result, list)