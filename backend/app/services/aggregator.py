"""
Verdict aggregation service.

Combines per-claim verdicts into an overall document-level verdict,
applying configurable weighting and conflict-resolution strategies.
"""
from __future__ import annotations

__all__ = ["Aggregator"]

class Aggregator:
    """Aggregate individual claim verdicts into a document verdict."""

    def aggregate(self, verdicts: list[dict]) -> dict:
        """Return a single aggregated verdict dict."""
        # TODO: implement
        pass
