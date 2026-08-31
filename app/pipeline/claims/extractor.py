"""
Claims Extractor — fact-checking pipeline module.

Usage:
    from app.pipeline.claims.extractor import extract_claims

    result = await extract_claims("Some text with factual claims...")
    print(result["selected_claims"])
    print(result["disambiguated"])
    print(result["decomposed"])
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.services.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_claims(text: str) -> dict:
    """
    Extract, disambiguate, and decompose factual claims from *text*.

    Makes exactly one LLM call (task_type='fast' via llm_client) and returns
    a plain dict with three keys: selected_claims, disambiguated, decomposed.

    Args:
        text: Raw input passage to analyse.

    Returns:
        dict with keys "selected_claims", "disambiguated", "decomposed".
        If parsing fails, returns {"selected_claims": [], "disambiguated": [], "decomposed": []}.
    """
    if not text or not text.strip():
        return {"selected_claims": [], "disambiguated": [], "decomposed": []}

    prompt = (
        "Respond ONLY with valid JSON. No explanation, no markdown, no code fences.\n"
        '{"selected_claims": [...], "disambiguated": [...], "decomposed": [[...]]}'
    )

    try:
        raw = await call_llm(prompt, task_type="fast")
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {"selected_claims": [], "disambiguated": [], "decomposed": []}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("LLM response is not valid JSON: %s", exc)
        return {"selected_claims": [], "disambiguated": [], "decomposed": []}

    # Validate required keys
    required_keys = {"selected_claims", "disambiguated", "decomposed"}
    missing = required_keys - set(data.keys())
    if missing:
        logger.error("LLM response missing required keys: %s", missing)
        return {"selected_claims": [], "disambiguated": [], "decomposed": []}

    # Ensure all three lists have the same length
    sc = data.get("selected_claims", [])
    dis = data.get("disambiguated", [])
    dec = data.get("decomposed", [])

    if not (len(sc) == len(dis) == len(dec)):
        logger.error(
            "List length mismatch — selected_claims=%d, disambiguated=%d, decomposed=%d",
            len(sc), len(dis), len(dec),
        )
        return {"selected_claims": [], "disambiguated": [], "decomposed": []}

    logger.debug("Parsed %d claim(s) successfully.", len(sc))
    logger.info("Claims extracted | count=%d", len(sc))
    return {"selected_claims": sc, "disambiguated": dis, "decomposed": dec}


if __name__ == "__main__":
    import pprint

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    EXAMPLE_TEXT = (
        "Elon Musk founded Tesla in 2003 and it is now the world's most valuable "
        "car company with over 100 billion in revenue."
    )

    print("=" * 60)
    print("INPUT TEXT:")
    print(EXAMPLE_TEXT)
    print("=" * 60)

    claims = asyncio.run(extract_claims(EXAMPLE_TEXT))
    print("\nRESULT:")
    pprint.pprint(claims)