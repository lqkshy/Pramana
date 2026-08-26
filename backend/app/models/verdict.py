"""
Verdict model for Pramana.

A Verdict is the final judgement for a single Claim: one of
SUPPORTED, REFUTED, or NOT_ENOUGH_INFO, along with a confidence
score and a human-readable explanation.
"""
from __future__ import annotations
from enum import Enum

__all__ = ["VerdictLabel", "VerdictBase", "VerdictRead"]

class VerdictLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"

class VerdictBase:
    # TODO: implement
    pass

class VerdictRead(VerdictBase):
    pass
