"""tests/test_verification.py — mocks the LLM, no API keys."""
import json

import pytest

from app.models.schemas import EvidencePassage, VerdictLabel
from app.pipeline import verification


def _p(url, q):
    return EvidencePassage(text="evidence text " * 5, source_url=url, source_quality=q)


def _raw(**kw):
    base = {"verdict": "SUPPORTED", "confidence": 0.9, "explanation": "because [1]", "cited_sources": [1]}
    base.update(kw)
    return json.dumps(base)


def test_evidence_strength_uses_locked_formula():
    passages = [_p("https://a.gov", 0.9), _p("https://b.edu", 0.85), _p("https://c.com", 0.8)]
    r = verification.build_result("c", passages, _raw(cited_sources=[1, 2, 3]))
    assert r.evidence_strength == pytest.approx(0.85, abs=1e-3)  # avg 0.85 × min(1, 3/3)


def test_single_source_is_weak():
    r = verification.build_result("c", [_p("https://a.com", 0.8)], _raw())
    assert r.evidence_strength == pytest.approx(0.2667, abs=1e-3)


def test_same_url_cited_twice_counts_once():
    passages = [_p("https://a.com", 0.8), _p("https://a.com", 0.8)]
    r = verification.build_result("c", passages, _raw(cited_sources=[1, 2]))
    assert r.cited_source_urls == ["https://a.com"]


def test_invalid_citation_numbers_are_ignored_and_verdict_downgraded():
    r = verification.build_result("c", [_p("https://a.com", 0.8)], _raw(cited_sources=[7]))
    assert r.verdict == VerdictLabel.INSUFFICIENT_EVIDENCE
    assert r.evidence_strength == 0.0


def test_label_aliases_and_bad_json():
    r = verification.build_result("c", [_p("https://a.com", 0.8)], _raw(verdict="Refuted"))
    assert r.verdict == VerdictLabel.CONTRADICTED
    bad = verification.build_result("c", [_p("https://a.com", 0.8)], "not json")
    assert bad.verdict == VerdictLabel.INSUFFICIENT_EVIDENCE


async def test_no_passages_skips_llm(monkeypatch):
    async def boom(**kw):
        raise AssertionError("LLM must not be called without evidence")

    monkeypatch.setattr(verification, "call_llm", boom)
    r = await verification.verify_claim("c", [])
    assert r.verdict == VerdictLabel.INSUFFICIENT_EVIDENCE
