"""
Pydantic and SQLAlchemy models for a Claim entity.

A Claim represents one atomic, checkable statement extracted
from user-submitted text.  It tracks status through the pipeline
from PENDING → IN_PROGRESS → COMPLETE.
"""
from __future__ import annotations
from enum import Enum

__all__ = ["ClaimStatus", "ClaimBase", "ClaimCreate", "ClaimRead"]

class ClaimStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"

class ClaimBase:
    """Shared claim fields."""
    # TODO: implement
    pass

class ClaimCreate(ClaimBase):
    """Fields required when creating a new claim."""
    pass

class ClaimRead(ClaimBase):
    """Fields returned when reading a claim."""
    pass
