"""
Claim extraction service.

Sends raw user text to an LLM and parses the response into a list
of discrete, verifiable atomic claims.  Handles prompt construction,
response validation, and retry logic.
"""
from __future__ import annotations

__all__ = ["ClaimExtractor"]

class ClaimExtractor:
    """Extract atomic claims from free-form text using an LLM."""

    def __init__(self) -> None:
        # TODO: inject LLM client and prompt config
        pass

    async def extract(self, text: str) -> list[str]:
        """Return a list of atomic claim strings parsed from *text*."""
        # TODO: implement
        pass
