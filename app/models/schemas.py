"""
Pydantic v2 schemas for the fact-checking API.

Defines request/response models for claim extraction, verification results,
and shared data types used across the pipeline.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic import field_validator, model_validator


class VerdictEnum(str, Enum):
    """Verdict values for claim verification outcomes."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class ClaimInput(BaseModel):
    """Raw text payload to be fact-checked."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
    )

    text: str = Field(
        min_length=1,
        max_length=10_000,
        description="Raw input text to be fact-checked",
    )


class ExtractedClaim(BaseModel):
    """A single claim extracted from source text, with sub-claims."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
    )

    original: str = Field(
        min_length=1,
        description="Verbatim claim from source text",
    )
    disambiguated: str = Field(
        min_length=1,
        description="Rewritten to be self-contained",
    )
    sub_claims: list[str] = Field(
        min_length=1,
        description="Atomic sub-claims",
    )

    @field_validator("sub_claims", mode="after")
    @classmethod
    def _non_empty_sub_claims(cls, value: list[str]) -> list[str]:
        """Reject any element that is empty or whitespace-only after stripping."""
        for item in value:
            if not item or not item.strip():
                raise ValueError("sub_claims must not contain empty or whitespace-only entries")
        return value


class VerificationResult(BaseModel):
    """Outcome of verifying a single claim against evidence."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
    )

    claim: str = Field(
        min_length=1,
        description="The claim being verified",
    )
    verdict: VerdictEnum = Field(
        description="Fact-check outcome",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the verdict",
    )
    evidence_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality/strength of the evidence",
    )
    explanation: str = Field(
        min_length=1,
        description="Human-readable reasoning",
    )
    matched_claim_id: Optional[str] = Field(
        default=None,
        description="Optional reference ID",
    )

    @model_validator(mode="after")
    def _check_confidence_threshold(self) -> "VerificationResult":
        """Enforce minimum confidence for definitive verdicts."""
        if self.verdict == VerdictEnum.SUPPORTED and self.confidence < 0.5:
            raise ValueError("A SUPPORTED verdict requires confidence >= 0.5")
        if self.verdict == VerdictEnum.CONTRADICTED and self.confidence < 0.5:
            raise ValueError("A CONTRADICTED verdict requires confidence >= 0.5")
        return self


class VerifyResponse(BaseModel):
    """Aggregated verification results for a set of claims."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
    )

    claims: list[VerificationResult] = Field(
        min_length=1,
        description="All verification results",
    )