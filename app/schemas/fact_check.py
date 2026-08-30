"""
Pydantic v2 schemas for the fact-checking API.

Defines request/response models shared between the extraction, verification,
and API layers. All models are immutable (frozen) and strip surrounding
whitespace on string fields.
"""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ClaimInput(BaseModel):
    """Raw text payload to be fact-checked."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
        json_schema_extra={
            "text": "Elon Musk founded Tesla in 2003 and it is now the world's most "
            "valuable car company with over 100 billion in revenue."
        },
    )

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=10_000,
            description="Raw input text to be fact-checked",
        ),
    ]


class ExtractedClaim(BaseModel):
    """A single claim extracted from source text, with sub-claims."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
        json_schema_extra={
            "original": "Tesla was founded in 2003",
            "disambiguated": "Tesla, the electric vehicle company, was founded in the year 2003",
            "sub_claims": [
                "Tesla is an electric vehicle company",
                "Tesla was founded in 2003",
            ],
        },
    )

    original: Annotated[
        str,
        Field(min_length=1, description="Verbatim claim from source text"),
    ]
    disambiguated: Annotated[
        str,
        Field(min_length=1, description="Rewritten to be self-contained"),
    ]
    sub_claims: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Atomic sub-claims",
        ),
    ]

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
        json_schema_extra={
            "claim": "Tesla was founded in 2003",
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "evidence_strength": 0.88,
            "explanation": "Multiple reliable sources confirm Tesla's incorporation date as July 2003.",
            "matched_claim_id": "claim_001",
        },
    )

    claim: Annotated[
        str,
        Field(min_length=1, description="The claim being verified"),
    ]
    verdict: Annotated[
        Literal[
            "SUPPORTED",
            "CONTRADICTED",
            "INSUFFICIENT_EVIDENCE",
            "CONFLICTING_EVIDENCE",
        ],
        Field(description="Fact-check outcome"),
    ]
    confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Model confidence in the verdict"),
    ]
    evidence_strength: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Quality/strength of the evidence"),
    ]
    explanation: Annotated[
        str,
        Field(min_length=1, description="Human-readable reasoning"),
    ]
    matched_claim_id: Annotated[
        str | None,
        Field(default=None, description="Optional reference ID"),
    ]

    @model_validator(mode="after")
    def _check_confidence_threshold(self) -> "VerificationResult":
        """Enforce minimum confidence for definitive verdicts."""
        if self.verdict == "SUPPORTED" and self.confidence < 0.5:
            raise ValueError("A SUPPORTED verdict requires confidence >= 0.5")
        if self.verdict == "CONTRADICTED" and self.confidence < 0.5:
            raise ValueError("A CONTRADICTED verdict requires confidence >= 0.5")
        return self


class VerifyResponse(BaseModel):
    """Aggregated verification results for a set of claims."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
        json_schema_extra={
            "claims": [
                {
                    "claim": "Tesla was founded in 2003",
                    "verdict": "SUPPORTED",
                    "confidence": 0.95,
                    "evidence_strength": 0.88,
                    "explanation": "Multiple reliable sources confirm Tesla's incorporation date as July 2003.",
                    "matched_claim_id": "claim_001",
                }
            ]
        },
    )

    claims: Annotated[
        list[VerificationResult],
        Field(min_length=1, description="All verification results"),
    ]


__all__ = [
    "ClaimInput",
    "ExtractedClaim",
    "VerificationResult",
    "VerifyResponse",
]


if __name__ == "__main__":
    # --- ClaimInput ---
    ci = ClaimInput(text="  Tesla was founded in 2003.  ")
    assert ci.text == "Tesla was founded in 2003.", f"strip failed: {ci.text!r}"

    # --- ExtractedClaim ---
    ec = ExtractedClaim(
        original="Tesla was founded in 2003",
        disambiguated="Tesla, the EV company, was founded in 2003",
        sub_claims=["Tesla is an EV company", "Tesla was founded in 2003"],
    )
    assert len(ec.sub_claims) == 2

    # --- VerificationResult: valid SUPPORTED ---
    vr = VerificationResult(
        claim="Tesla was founded in 2003",
        verdict="SUPPORTED",
        confidence=0.95,
        evidence_strength=0.88,
        explanation="Multiple sources confirm July 2003.",
        matched_claim_id="claim_001",
    )
    assert vr.verdict == "SUPPORTED"

    # --- VerificationResult: valid INSUFFICIENT_EVIDENCE low confidence ---
    vr2 = VerificationResult(
        claim="Tesla is the most valuable car company",
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.3,
        evidence_strength=0.2,
        explanation="Insufficient public data found.",
    )
    assert vr2.matched_claim_id is None

    # --- VerifyResponse ---
    resp = VerifyResponse(claims=[vr, vr2])
    assert len(resp.claims) == 2

    # --- Validation errors fire correctly ---
    import sys

    try:
        ClaimInput(text="")
        print("FAIL: empty text should be rejected")
        sys.exit(1)
    except ValidationError:
        pass

    try:
        ExtractedClaim(
            original="x", disambiguated="y", sub_claims=["valid", "   "]
        )
        print("FAIL: whitespace-only sub_claim should be rejected")
        sys.exit(1)
    except ValidationError:
        pass

    try:
        VerificationResult(
            claim="x", verdict="SUPPORTED", confidence=0.2,
            evidence_strength=0.5, explanation="low conf"
        )
        print("FAIL: SUPPORTED + confidence<0.5 should be rejected")
        sys.exit(1)
    except ValidationError:
        pass

    print("All assertions passed.")
