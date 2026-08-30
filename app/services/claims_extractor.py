"""
Claims Extractor — fact-checking pipeline module.

Selects check-worthy factual claims from a passage, rewrites them to be
self-contained and unambiguous, and decomposes each into atomic sub-claims
via exactly one LLM call (task_type="fast").

Usage:
    from app.services.claims_extractor import extract_claims
    result = extract_claims("Some text with factual claims...")
    print(result.selected_claims)
    print(result.disambiguated)
    print(result.decomposed)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)


class ClaimsExtractionError(Exception):
    """Raised when the LLM response cannot be parsed into a valid ClaimsResult."""


@dataclass(frozen=True)
class ClaimsResult:
    """Immutable container for the three parallel claim lists."""
    selected_claims: list[str]
    disambiguated: list[str]
    decomposed: list[list[str]]


_PROMPT = """\
You are a fact-checking assistant. Given the passage below, extract factual claims worth verifying.

Return ONLY a valid JSON object with EXACTLY these 3 keys and no other text, markdown, or explanation:

{
  "selected_claims": [
    "<verbatim or near-verbatim factual sentence from the passage>"
  ],
  "disambiguated": [
    "<the same claim rewritten to be fully self-contained, resolving all pronouns and references>"
  ],
  "decomposed": [
    ["<atomic sub-claim 1>", "<atomic sub-claim 2>", "..."]
  ]
}

Rules:
- selected_claims, disambiguated, and decomposed must all have the SAME length.
- decomposed[i] contains the atomic sub-claims for disambiguated[i].
- Skip opinions, predictions, and non-verifiable statements.
- If there are no checkable claims, return empty lists for all three keys.

### EXAMPLE INPUT
"Elon Musk, who was born in South Africa in 1971, founded SpaceX in 2002 and the company launched its first Falcon 9 rocket in 2010."

### EXAMPLE OUTPUT
{
  "selected_claims": [
    "Elon Musk was born in South Africa in 1971",
    "Elon Musk founded SpaceX in 2002",
    "SpaceX launched its first Falcon 9 rocket in 2010"
  ],
  "disambiguated": [
    "Elon Musk was born in Pretoria, South Africa in 1971",
    "Elon Musk founded the aerospace company SpaceX in 2002",
    "SpaceX, the aerospace company founded by Elon Musk, launched its first Falcon 9 rocket in the year 2010"
  ],
  "decomposed": [
    ["Elon Musk was born in South Africa", "Elon Musk's birth year is 1971"],
    ["Elon Musk founded SpaceX", "SpaceX was founded in 2002"],
    ["SpaceX launched a Falcon 9 rocket", "The Falcon 9 launch was SpaceX's first", "The launch occurred in 2010"]
  ]
}

### PASSAGE TO ANALYSE
{text}
"""


def _build_prompt(text: str) -> str:
    """Return the LLM prompt with the passage substituted for {text}."""
    return _PROMPT.replace("{text}", text)


def _parse_response(raw: str) -> ClaimsResult:
    """Parse the raw LLM string into a ClaimsResult, handling all failure modes."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ClaimsExtractionError(
            f"LLM response is not valid JSON: {exc}\n"
            f"Raw response (first 500 chars):\n{raw[:500]}"
        ) from exc

    required = {"selected_claims", "disambiguated", "decomposed"}
    missing = required - data.keys()
    if missing:
        raise ClaimsExtractionError(f"LLM response missing required keys: {missing}")

    sc, dis, dec = data["selected_claims"], data["disambiguated"], data["decomposed"]
    if not (len(sc) == len(dis) == len(dec)):
        raise ClaimsExtractionError(
            f"List length mismatch — selected_claims={len(sc)}, "
            f"disambiguated={len(dis)}, decomposed={len(dec)}. "
            "All three must have the same length."
        )

    logger.debug("Parsed %d claim(s) successfully.", len(sc))
    return ClaimsResult(selected_claims=sc, disambiguated=dis, decomposed=dec)


def extract_claims(text: str) -> ClaimsResult:
    """Extract, disambiguate, and decompose factual claims from *text* (one LLM call)."""
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    prompt = _build_prompt(text)
    raw = asyncio.run(call_llm(prompt, task_type="fast"))  # exactly ONE Groq call
    return _parse_response(raw)
