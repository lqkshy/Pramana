"""
Evidence model for Pramana.

Each Evidence record links a retrieved source (URL, snippet, metadata)
to the Claim it was fetched to support or refute.
"""
from __future__ import annotations

__all__ = ["EvidenceBase", "EvidenceCreate", "EvidenceRead"]

class EvidenceBase:
    # TODO: implement
    pass

class EvidenceCreate(EvidenceBase):
    pass

class EvidenceRead(EvidenceBase):
    pass
