"""Unit tests for Stage 1 response parsing (no API calls)."""

import json

from app.pipeline.claims.extractor import _parse_response


def test_parse_valid_response():
    raw = json.dumps({
        "selected_sentences": ["Tesla was founded in 2003 and is based in Texas."],
        "disambiguated": ["Tesla was founded in 2003 and Tesla is based in Texas."],
        "decomposed": ["Tesla was founded in 2003.", "Tesla is based in Texas."],
    })
    result = _parse_response(raw, "original text")
    assert [c.text for c in result.claims] == ["Tesla was founded in 2003.", "Tesla is based in Texas."]
    assert result.raw_input == "original text"


def test_parse_strips_markdown_fences():
    raw = '```json\n{"selected_sentences": [], "disambiguated": [], "decomposed": ["A fact."]}\n```'
    assert len(_parse_response(raw, "x").claims) == 1


def test_parse_garbage_returns_empty_result():
    result = _parse_response("this is not json", "x")
    assert result.claims == []


def test_empty_decomposed_claims_are_skipped():
    raw = '{"selected_sentences": ["s"], "disambiguated": ["s"], "decomposed": ["", "  ", "Real claim."]}'
    assert [c.text for c in _parse_response(raw, "x").claims] == ["Real claim."]
