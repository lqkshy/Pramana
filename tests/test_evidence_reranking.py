"""tests/test_evidence_reranking.py — no API keys, no model download."""
import pytest

from app.models.schemas import EvidencePassage, ScoredSource
from app.pipeline import evidence, reranking


@pytest.fixture(autouse=True)
def no_cross_encoder(monkeypatch):
    monkeypatch.setattr(reranking, "_get_cross_encoder", lambda: None)  # use overlap fallback


def _src(url, text, q=0.8):
    return ScoredSource(url=url, snippet=text, raw_text=text, quality_score=q)


def test_chunk_text_respects_max_words():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(100))
    chunks = reranking.chunk_text(text, max_words=50)
    assert chunks and all(len(c.split()) <= 50 for c in chunks)


async def test_rerank_picks_relevant_and_limits_per_source():
    claim = "NASA launched Artemis I in November 2022."
    good = "NASA launched the Artemis I mission on November 16 2022 from Kennedy Space Center in Florida."
    bad = "The recipe calls for two cups of flour and a pinch of salt for the dough."
    sources = [_src("https://a.com", good + " " + good), _src("https://b.com", bad)]
    out = await reranking.rerank_passages(claim, sources, top_k=3, max_per_source=1)
    assert out[0].source_url == "https://a.com"
    assert len({p.source_url for p in out}) == len(out)


async def test_refine_rejects_hallucinated_sentence(monkeypatch):
    passages = [EvidencePassage(text="Paris is the capital of France and has 2 million people.",
                                source_url="https://a.com", source_quality=0.8)]

    async def fake_llm(**kw):
        return '{"evidence": [{"id": 1, "sentence": "Paris has 90 million people."}]}'

    monkeypatch.setattr(evidence, "call_llm", fake_llm)
    out = await evidence.refine_passages("Paris population", passages)
    assert out[0].text == passages[0].text   # hallucination rejected


async def test_refine_accepts_verbatim_sentence(monkeypatch):
    passages = [EvidencePassage(text="Intro text here. Paris has 2 million people. Other stuff.",
                                source_url="https://a.com", source_quality=0.8)]

    async def fake_llm(**kw):
        return '{"evidence": [{"id": 1, "sentence": "Paris has 2 million people."}]}'

    monkeypatch.setattr(evidence, "call_llm", fake_llm)
    out = await evidence.refine_passages("Paris population", passages)
    assert out[0].text == "Paris has 2 million people."


def test_html_to_text_strips_boilerplate():
    html = "<nav>menu</nav><p>" + "word " * 12 + "</p><script>x()</script><footer>foot</footer>"
    text = evidence.html_to_text(html)
    assert "menu" not in text and "foot" not in text and "word" in text
