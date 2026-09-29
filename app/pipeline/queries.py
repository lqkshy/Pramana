"""
app/pipeline/queries.py
────────────────────────
Stage 4 of the Pramana pipeline.

Generate 2–3 targeted search queries for a single atomic claim.
One Groq call (task_type="fast" → Llama 3.1 8B).

Benchmark mode: This stage is NOT called when USE_LIVE_SEARCH=false.
  - averitec_loader.py and scifact_loader.py supply pre-retrieved evidence.
  - retrieval.py (Stage 5) returns it directly, bypassing Steps 4 and 5.

UI mode (USE_LIVE_SEARCH=true): This stage generates queries → retrieval.py feeds them to Tavily.

The prompt asks for:
  - Diverse angles (not three ways to say the same thing)
  - Named entities + date context when present
  - One broader "context" query and one narrow "specific fact" query
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.models.schemas import QueryGenerationResult
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a search query generator for a fact-checking pipeline. "
    "Your job is to generate 2–3 diverse, targeted web search queries "
    "that will help retrieve evidence to verify a factual claim. "
    "Return ONLY valid JSON with no extra text, preamble, or markdown fences."
)

_PROMPT_TEMPLATE = """Generate 2–3 search queries to find evidence for or against this claim.

Claim: {claim}

Requirements:
- Queries must be diverse — cover different angles, not rephrasing of the same query.
- Include key named entities, organisations, dates, and statistics from the claim.
- One query should be broad (context / background), one should be specific (the exact fact).
- If the claim includes a number or statistic, write one query targeting that specific figure.
- Keep each query under 12 words.
- Do NOT include "fact check" or "is it true" in the queries — they bias results.

Return ONLY this JSON object:
{{
  "queries": [
    "first search query here",
    "second search query here"
  ]
}}

You may include a third query if the claim has multiple independent facts to verify.
Maximum 3 queries. Minimum 2 queries."""


# ── Public API ─────────────────────────────────────────────────────────────────
async def generate_queries(claim: str) -> QueryGenerationResult:
    """
    Generate 2–3 targeted search queries for a single atomic claim.

    Args:
        claim: A check-worthy atomic claim from Stage 2.

    Returns:
        QueryGenerationResult with a list of 2–3 query strings.
    """
    prompt = _PROMPT_TEMPLATE.format(claim=claim.strip())

    raw = await call_llm(
        prompt=prompt,
        task_type="fast",
        system_prompt=_SYSTEM,
        max_tokens=256,
        temperature=0.2,   # Slight variation to get diverse queries, not pure determinism.
        json_mode=True,
    )

    queries = _parse_queries(raw, claim)

    result = QueryGenerationResult(claim=claim, queries=queries)

    logger.info(
        "generate_queries: claim=%r → %d queries: %s",
        claim[:60],
        len(queries),
        queries,
    )
    return result


async def generate_queries_batch(
    claims: list[str],
) -> list[QueryGenerationResult]:
    """
    Generate queries for a list of claims sequentially.

    Sequential (not parallel) to respect Groq RPM limits.
    The rate limiter in llm_client.py handles burst protection automatically.

    Args:
        claims: List of check-worthy atomic claims.

    Returns:
        List of QueryGenerationResult in the same order as input claims.
    """
    results: list[QueryGenerationResult] = []
    for claim in claims:
        result = await generate_queries(claim)
        results.append(result)
    return results


# ── Response parsing ───────────────────────────────────────────────────────────
def _parse_queries(raw: str, claim: str) -> list[str]:
    """
    Parse the LLM response into a list of 2–3 query strings.

    Graceful degradation:
      - Malformed JSON → fallback to extracting quoted strings
      - Still no queries found → return two naive keyword-extraction queries
    """
    # Clean markdown fences if present.
    cleaned = _strip_fences(raw)

    # Try JSON parse.
    try:
        data = json.loads(cleaned)
        queries: list[str] = data.get("queries", [])
        if isinstance(queries, list) and len(queries) >= 2:
            # Enforce 2–3 limit and strip whitespace.
            return [str(q).strip() for q in queries[:3] if str(q).strip()]
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "generate_queries: JSON parse failed for claim=%r. Raw: %r",
            claim[:60],
            raw[:300],
        )

    # Fallback 1: extract quoted strings from the raw text.
    quoted = re.findall(r'"([^"]{5,})"', raw)
    if len(quoted) >= 2:
        logger.debug("generate_queries: extracted %d quoted strings as fallback", len(quoted))
        return quoted[:3]

    # Fallback 2: naive keyword extraction from the claim itself.
    logger.warning(
        "generate_queries: using keyword fallback for claim=%r", claim[:60]
    )
    return _keyword_fallback(claim)


def _keyword_fallback(claim: str) -> list[str]:
    """
    Last-resort query generation: two naive queries derived from the claim text.
    Used only when the LLM response is completely unparseable.
    """
    # Query 1: first 10 words of the claim
    words = claim.split()
    q1 = " ".join(words[:10])

    # Query 2: last 10 words of the claim (different angle)
    q2 = " ".join(words[-10:]) if len(words) > 10 else " ".join(words)

    # Deduplicate
    if q1 == q2:
        return [q1, f"{q1} evidence source"]
    return [q1, q2]


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text
