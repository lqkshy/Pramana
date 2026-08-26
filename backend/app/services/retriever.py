"""
Evidence retrieval service.

Given an atomic claim, fetches supporting or contradicting evidence
from one or more sources (web search API, internal vector store, etc.)
and returns a ranked list of Evidence objects.
"""
from __future__ import annotations

__all__ = ["Retriever"]

class Retriever:
    """Retrieve evidence for a single atomic claim."""

    async def retrieve(self, claim: str) -> list[dict]:
        """Return a list of evidence dicts for the given claim string."""
        # TODO: implement
        pass
