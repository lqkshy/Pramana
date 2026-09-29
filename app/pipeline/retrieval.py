"""
app/pipeline/retrieval.py
──────────────────────────
Stage 5 of the Pramana pipeline.

Two modes — controlled by USE_LIVE_SEARCH environment variable:

  USE_LIVE_SEARCH=true  (FastAPI server)
      → Call Tavily API with generated queries
      → Used for real user UI submissions (1,000 credits/month quota)

  USE_LIVE_SEARCH=false (eval scripts — DEFAULT in .env)
      → Accept pre_retrieved_evidence argument and return it directly
      → Used by averitec_loader.py and scifact_loader.py
      → Zero Tavily credits consumed during benchmarks

Quota protection (v3.1 Fix 13):
  AVeriTeC:  500 samples × 2 queries = 1,000 Tavily calls AVOIDED
  SciFact:  1,400 samples × 2 queries = 2,800 Tavily calls AVOIDED
  Monthly quota stays intact for real user submissions.

The pipeline structure is identical in both modes.
Steps 6–9 receive the same RetrievalResult type regardless of which path ran.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from app.models.schemas import RetrievalResult, RetrievedSource

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# USE_LIVE_SEARCH controls which code path runs.
# Default: false (safe for development and all eval scripts).
# FastAPI server overrides this to true for real user submissions.
USE_LIVE_SEARCH: bool = os.getenv("USE_LIVE_SEARCH", "false").lower() == "true"

# Number of results to request from Tavily per query.
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# Concurrency limit for parallel Tavily queries (stay within API limits).
_TAVILY_SEMAPHORE = asyncio.Semaphore(3)


# ── Public API ─────────────────────────────────────────────────────────────────
async def retrieve_evidence(
    claim: str,
    queries: list[str],
    pre_retrieved_evidence: Optional[list[dict]] = None,
) -> RetrievalResult:
    """
    Retrieve evidence sources for a claim.

    Two code paths depending on USE_LIVE_SEARCH:

    Live mode (USE_LIVE_SEARCH=true):
        - Calls Tavily API with the provided queries
        - Returns 5–8 unique source URLs with titles and snippets
        - Used by FastAPI /verify endpoint for real user submissions

    Eval mode (USE_LIVE_SEARCH=false):
        - Returns pre_retrieved_evidence directly (no Tavily call)
        - pre_retrieved_evidence is supplied by averitec_loader / scifact_loader
        - Must be a list of dicts with at least {"url": ..., "snippet": ...}

    Args:
        claim:                   The atomic claim being verified.
        queries:                 2–3 search queries from Stage 4.
        pre_retrieved_evidence:  Pre-loaded evidence for eval mode.
                                 Required when USE_LIVE_SEARCH=false and the
                                 caller has already loaded evidence from a dataset.

    Returns:
        RetrievalResult with sources list and mode indicator.
    """
    # Read USE_LIVE_SEARCH at call time (not at import time) so it can be
    # overridden per-call in tests or per-script in eval loaders.
    live_search = _is_live_search()

    if not live_search:
        return _use_pre_retrieved(claim, pre_retrieved_evidence or [])

    return await _call_tavily(claim, queries)


def set_live_search(enabled: bool) -> None:
    """
    Override USE_LIVE_SEARCH at runtime (used by eval scripts).

    averitec_loader.py and scifact_loader.py call this at startup:
        retrieval.set_live_search(False)

    The FastAPI server never calls this — it relies on the env var being true.
    """
    os.environ["USE_LIVE_SEARCH"] = "true" if enabled else "false"
    logger.info("USE_LIVE_SEARCH overridden to %s", enabled)


def _is_live_search() -> bool:
    """Read USE_LIVE_SEARCH from the environment at call time."""
    return os.getenv("USE_LIVE_SEARCH", "false").lower() == "true"


# ── Pre-retrieved path (benchmark eval) ────────────────────────────────────────
def _use_pre_retrieved(
    claim: str,
    evidence_dicts: list[dict],
) -> RetrievalResult:
    """
    Wrap pre-retrieved evidence in the RetrievalResult schema.
    No API calls. No quota usage.

    Each dict in evidence_dicts should have at minimum:
      { "url": str, "snippet": str }
    Optional keys: "title", "raw_text"
    """
    sources: list[RetrievedSource] = []
    for item in evidence_dicts:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        if not url:
            # Allow snippet-only items (e.g. SciFact abstract with no URL).
            # Assign a synthetic URL so downstream stages can track the source.
            url = f"pre_retrieved://item_{len(sources)}"

        sources.append(
            RetrievedSource(
                url=url,
                title=item.get("title", ""),
                snippet=item.get("snippet", item.get("text", "")),
                raw_text=item.get("raw_text", item.get("text", "")),
            )
        )

    logger.debug(
        "retrieve_evidence: pre_retrieved mode — %d sources for claim=%r",
        len(sources),
        claim[:60],
    )
    return RetrievalResult(claim=claim, sources=sources, mode="pre_retrieved")


# ── Tavily live search path ─────────────────────────────────────────────────────
async def _call_tavily(claim: str, queries: list[str]) -> RetrievalResult:
    """
    Call the Tavily search API for each query and merge results.

    - Queries run in parallel (bounded by _TAVILY_SEMAPHORE).
    - Duplicate URLs are deduplicated (keep first occurrence).
    - Returns up to TAVILY_MAX_RESULTS × len(queries) unique sources.
    """
    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set but USE_LIVE_SEARCH=true. "
            "Add TAVILY_API_KEY to your .env file."
        )

    if not queries:
        logger.warning("retrieve_evidence: no queries provided, returning empty result.")
        return RetrievalResult(claim=claim, sources=[], mode="live")

    tasks = [_tavily_single_query(q) for q in queries]
    per_query_results: list[list[RetrievedSource]] = await asyncio.gather(*tasks)

    # Merge and deduplicate by URL.
    seen_urls: set[str] = set()
    merged: list[RetrievedSource] = []
    for batch in per_query_results:
        for source in batch:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                merged.append(source)

    logger.info(
        "retrieve_evidence: Tavily returned %d unique sources for claim=%r",
        len(merged),
        claim[:60],
    )
    return RetrievalResult(claim=claim, sources=merged, mode="live")


async def _tavily_single_query(query: str) -> list[RetrievedSource]:
    """
    Execute a single Tavily search query.
    Uses a semaphore to limit concurrent requests.
    """
    async with _TAVILY_SEMAPHORE:
        try:
            return await _tavily_request(query)
        except Exception as exc:
            logger.warning(
                "Tavily query failed: query=%r error=%s", query[:80], exc
            )
            return []


async def _tavily_request(query: str) -> list[RetrievedSource]:
    """
    Make the actual HTTP request to the Tavily /search endpoint.

    Tavily API reference: https://docs.tavily.com/docs/python-sdk/tavily-search/api-reference
    """
    try:
        import httpx  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install httpx") from exc

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",      # "advanced" uses 2 credits; "basic" uses 1
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": False,  # Raw content fetched later in Stage 7 (evidence.py)
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

    sources: list[RetrievedSource] = []
    for result in data.get("results", []):
        url = result.get("url", "").strip()
        if not url:
            continue
        sources.append(
            RetrievedSource(
                url=url,
                title=result.get("title", ""),
                snippet=result.get("content", ""),
            )
        )

    logger.debug("Tavily query=%r → %d results", query[:60], len(sources))
    return sources
