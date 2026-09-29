"""tests/test_source_trust.py — pure Python, no API keys."""
from app.models.schemas import RetrievedSource
from app.services.source_trust import score_and_filter, score_url


def test_tiers():
    assert score_url("https://www.cdc.gov/x")[0] == 0.90
    assert score_url("https://mit.edu/a")[0] == 0.85
    assert score_url("https://www.reuters.com/a")[0] == 0.80
    assert score_url("https://en.wikipedia.org/wiki/X")[0] == 0.75
    assert score_url("https://random-blog.net")[0] == 0.40
    assert score_url("https://infowars.com/x")[0] == 0.10


def test_pre_retrieved_urls_are_not_dropped():
    """Regression: benchmark mode uses pre_retrieved:// URLs; they must survive filtering."""
    sources = [RetrievedSource(url="pre_retrieved://item_0", snippet="abstract text")]
    assert len(score_and_filter(sources)) == 1
