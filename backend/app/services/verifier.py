"""
Claim verification service.

Takes a claim and a set of evidence passages and asks an LLM to
produce a structured verdict (label + confidence + explanation).
"""
from __future__ import annotations

__all__ = ["Verifier"]

class Verifier:
    """Verify a single claim against retrieved evidence."""

    async def verify(self, claim: str, evidence: list[dict]) -> dict:
        """Return a verdict dict with keys: label, confidence, explanation."""
        # TODO: implement
        pass
