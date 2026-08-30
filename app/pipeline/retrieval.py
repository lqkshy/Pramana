"""Retrieval module for the fact-checking pipeline.

Handles evidence retrieval either via live search (Tavily) or a stub mode
that returns pre-provided evidence.
"""

import asyncio
import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _is_live_search_enabled() -> bool:
    """Return True when USE_LIVE_SEARCH env var is set to 'true'."""
    return os.getenv("USE_LIVE_SEARCH", "false").strip().lower() == "true"


class TavilyClient:
    """Minimal stand-in for tavily-python when the package is unavailable.

    The real tavily-python provides `TavilyClient` with a `search` method;
    this dummy mirrors that interface so the rest of the pipeline can import
    it without the dependency installed.
    """

    def search(self, query: str, max_results: int = 5):  # type: ignore[override]
        # In a real installation this would call the Tavily API.
        # For this module we raise so the __main__ block can demo the
        # "live" path failing gracefully if the key is missing.
        raise NotImplementedError("TavilyClient.search not implemented – set TAVILY_API_KEY or USE_LIVE_SEARCH=false")


async def retrieve_evidence(query: str, pre_retrieved_evidence: list = None) -> list[dict]:
    """Retrieve evidence for a factual query.

    Args:
        query: The user's factual question or claim.
        pre_retrieved_evidence: Optional list of dicts to return directly
            when live search is disabled.

    Returns:
        A list of dicts, each containing at least "url", "title", and
        "content" keys. Returns an empty list when no evidence can be found.
    """
    if _is_live_search_enabled():
        logger.info("LIVE SEARCH enabled: using TavilyClient")
        try:
            from tavily import TavilyClient as _TavilyClient
        except ImportError:
            logger.error("tavily package not installed – cannot perform live search")
            return []

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY not set – cannot perform live search")
            return []

        client = _TavilyClient(api_key=api_key)
        try:
            resp = client.search(query=query, max_results=5)
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return []

        # Normalise Tavily response into the expected dict shape
        results = resp.get("results", [])
        normalised = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
            }
            for r in results
        ]
        return normalised

    else:
        if pre_retrieved_evidence is not None:
            logger.info("LIVE SEARCH disabled: returning pre-retrieved evidence (%d items)", len(pre_retrieved_evidence))
            return pre_retrieved_evidence

        logger.warning("LIVE SEARCH disabled and no pre-retrieved evidence – returning empty list")
        return []


if __name__ == "__main__":
    # Quick test: force live-search mode for demonstration
    import os as _os

    _os.environ["USE_LIVE_SEARCH"] = "true"
    # Ensure TAVILY_API_KEY is present; if not, the function will log and return []
    _os.environ.setdefault("TAVILY_API_KEY", "")

    results = asyncio.run(retrieve_evidence("Is the Earth flat?"))

    print("\nTop 3 results:")
    for r in results[:3]:
        print(f"  - {r.get('title', 'Untitled')} ({r.get('url', 'no-url')})")