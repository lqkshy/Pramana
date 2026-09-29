"""API tests — /health and a fully mocked end-to-end /verify run (no keys, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ClaimMatchResult, RetrievedSource

client = TestClient(app)


def test_health_returns_ok():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "provider" in body and "dev_mode" in body


def test_verify_rejects_empty_text():
    assert client.post("/verify", json={"text": "   "}).status_code == 422


CLAIM = "NASA launched the Artemis I mission on November 16, 2022."


async def _fake_llm(prompt, task_type="fast", system_prompt=None, **kwargs):
    """Routes on the system prompt so each pipeline stage gets a plausible reply."""
    s = system_prompt or ""
    if "claim extraction engine" in s:
        return json.dumps({"selected_sentences": [CLAIM], "disambiguated": [CLAIM], "decomposed": [CLAIM]})
    if "check-worthiness classifier" in s:
        return json.dumps({"is_check_worthy": True, "reason": "Specific, dated, falsifiable."})
    if "search query generator" in s:
        return json.dumps({"queries": ["NASA Artemis I launch date", "Artemis I November 2022 launch"]})
    if "extract evidence" in s:
        return json.dumps({"evidence": []})            # refinement is optional → keeps passages
    if "fact-checking analyst" in s:
        return json.dumps({
            "verdict": "SUPPORTED", "confidence": 0.93,
            "explanation": "Both [1] and [2] state the launch was on 16 November 2022.",
            "cited_sources": [1, 2],
        })
    raise AssertionError(f"unexpected LLM call: {s[:60]}")


@pytest.fixture
def mocked_pipeline(monkeypatch):
    import app.main as main_mod
    from app.pipeline import check_worthiness, evidence, queries, reranking, retrieval, verification
    from app.pipeline.claims import extractor

    for mod in (extractor, check_worthiness, queries, evidence, verification):
        monkeypatch.setattr(mod, "call_llm", _fake_llm)

    async def fake_tavily(query):
        return [
            RetrievedSource(url="https://www.nasa.gov/artemis-i", title="Artemis I",
                            snippet="NASA launched the Artemis I mission on November 16, 2022 from Kennedy Space Center in Florida."),
            RetrievedSource(url="https://www.reuters.com/artemis", title="Reuters",
                            snippet="The uncrewed Artemis I rocket lifted off on November 16, 2022 carrying the Orion spacecraft toward the Moon."),
            RetrievedSource(url="https://infowars.com/moon-hoax", title="Hoax",
                            snippet="Artemis is fake."),
        ]

    async def no_fetch(url):
        return ""

    async def no_match(claim):
        return ClaimMatchResult(claim=claim, matched=False)

    stored = []

    async def fake_store(claim_id, claim, result):
        stored.append(result)

    monkeypatch.setenv("USE_LIVE_SEARCH", "true")
    monkeypatch.setattr(retrieval, "TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(retrieval, "_tavily_request", fake_tavily)
    monkeypatch.setattr(evidence, "_fetch_text", no_fetch)
    monkeypatch.setattr(reranking, "_get_cross_encoder", lambda: None)
    monkeypatch.setattr(main_mod, "match_claim", no_match)
    monkeypatch.setattr(main_mod, "store_claim", fake_store)
    return stored


def test_verify_full_pipeline_mocked(mocked_pipeline):
    resp = client.post("/verify", json={"text": CLAIM})
    assert resp.status_code == 200
    body = resp.json()

    assert body["claims_found"] == 1
    result = body["results"][0]
    assert result["verdict"] == "SUPPORTED"
    assert set(result["cited_source_urls"]) == {
        "https://www.nasa.gov/artemis-i", "https://www.reuters.com/artemis"
    }
    # evidence_strength = avg(0.90, 0.80) × min(1, 2/3) = 0.5667  (locked v3.1 formula)
    assert result["evidence_strength"] == pytest.approx(0.5667, abs=1e-3)
    assert len(mocked_pipeline) == 1                      # Stage 9 stored the verdict


def test_verify_survives_a_failing_claim(mocked_pipeline, monkeypatch):
    """One claim blowing up must not 500 the whole request."""
    from app.pipeline import retrieval

    async def boom(query):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(retrieval, "_tavily_request", boom)
    body = client.post("/verify", json={"text": CLAIM}).json()
    assert body["results"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
