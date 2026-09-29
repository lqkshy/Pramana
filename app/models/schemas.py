"""
app/models/schemas.py
─────────────────────
Pydantic v2 data models for the Pramana pipeline.

evidence_strength formula (locked in v3.1 — do not change):
    evidence_strength = avg(quality_score of sources LLM cited as supporting)
                        × min(1.0, cited_source_count / 3)

  0.00 → LLM cited zero sources (INSUFFICIENT_EVIDENCE or hallucination)
  0.27 → 1 source @ quality 0.80  (0.80 × 1/3 = 0.267 — weak)
  0.57 → 2 sources avg 0.85       (0.85 × 2/3 = 0.567 — moderate)
  0.80 → 3+ sources avg 0.80      (0.80 × 1.0 = 0.80 — strong)

Computed AFTER the verification LLM call returns its cited_source_urls.
Look up each URL's quality_score from Step 6 (source_trust.py) and apply the formula.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator


# ── Verdict labels ─────────────────────────────────────────────────────────────
class VerdictLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


# ── Step 1 — Claim Extraction ──────────────────────────────────────────────────
class AtomicClaim(BaseModel):
    """A single atomic, verifiable sub-claim extracted from the input text."""

    text: str = Field(..., description="The atomic claim text, self-contained.")
    original_sentence: str = Field(
        ..., description="The source sentence from the input text."
    )
    disambiguated: str = Field(
        ...,
        description=(
            "Claim text with pronouns/ambiguities resolved using document context. "
            "Identical to `text` when no disambiguation is needed."
        ),
    )


class ExtractionResult(BaseModel):
    """Output of Stage 1 (claims/extractor.py). One structured Groq call."""

    selected_sentences: list[str] = Field(
        default_factory=list,
        description="Verifiable sentences chosen from the input.",
    )
    claims: list[AtomicClaim] = Field(
        default_factory=list,
        description="Decomposed atomic claims from selected sentences.",
    )
    raw_input: str = Field(default="", description="Original input text.")


# ── Step 2 — Check-worthiness ──────────────────────────────────────────────────
class CheckWorthinessResult(BaseModel):
    claim: str
    is_check_worthy: bool = Field(
        ...,
        description=(
            "True = specific, falsifiable, real-world assertion. "
            "False = opinion, prediction, rhetorical question, or vague generality."
        ),
    )
    reason: str = Field(default="", description="One-sentence justification from the model.")


# ── Step 3 — Claim Matching ────────────────────────────────────────────────────
class ClaimMatchResult(BaseModel):
    claim: str
    matched: bool = Field(default=False)
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_claim_id: Optional[str] = Field(
        default=None,
        description="ID of the previously verified claim if similarity > 0.92.",
    )
    cached_verdict: Optional["VerificationResult"] = Field(
        default=None,
        description="Full cached result to return immediately, skipping Steps 4–8.",
    )


# ── Step 4 — Query Generation ──────────────────────────────────────────────────
class QueryGenerationResult(BaseModel):
    claim: str
    queries: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="2–3 targeted search queries for this claim.",
    )


# ── Step 5 — Retrieval ─────────────────────────────────────────────────────────
class RetrievedSource(BaseModel):
    url: str
    title: str = Field(default="")
    snippet: str = Field(default="", description="Short excerpt from the page.")
    raw_text: str = Field(
        default="",
        description="Full scraped page text (populated in Step 7).",
    )


class RetrievalResult(BaseModel):
    claim: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    mode: str = Field(
        default="live",
        description="'live' = Tavily was called. 'pre_retrieved' = benchmark eval mode.",
    )


# ── Step 6 — Source Quality ────────────────────────────────────────────────────
class ScoredSource(BaseModel):
    url: str
    title: str = Field(default="")
    snippet: str = Field(default="")
    raw_text: str = Field(default="")
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic domain trust score. "
            ".gov=0.9 | .edu=0.85 | major_news=0.80 | wikipedia=0.75 | "
            ".org=0.60 | unknown=0.40 | disinfo=0.10"
        ),
    )
    trust_tier: str = Field(
        default="unknown",
        description="Human-readable tier label (e.g. 'government', 'academic', 'major_news').",
    )


# ── Step 7 — Evidence Extraction ───────────────────────────────────────────────
class EvidencePassage(BaseModel):
    text: str = Field(..., description="Top-3 re-ranked evidence passage text.")
    source_url: str
    source_quality: float = Field(ge=0.0, le=1.0)
    rerank_score: float = Field(
        default=0.0,
        description="Cross-encoder score — higher = more relevant to the claim.",
    )


# ── Step 8 — Verification ──────────────────────────────────────────────────────
class VerificationResult(BaseModel):
    """
    Full pipeline result for a single atomic claim.
    Returned by Stage 8 (verification.py) and stored in Supabase.
    """

    claim: str
    verdict: VerdictLabel
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in the verdict (0–1).",
    )
    explanation: str = Field(
        ...,
        description=(
            "Chain-of-thought explanation. Written for a non-technical reader. "
            "Should cite specific sources and explain the reasoning step by step."
        ),
    )
    evidence_passages: list[EvidencePassage] = Field(default_factory=list)
    cited_source_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs the LLM explicitly cited in its reasoning. "
            "Used to compute evidence_strength — must be a subset of evidence_passages URLs."
        ),
    )
    cited_source_qualities: list[float] = Field(
        default_factory=list,
        description=(
            "quality_score for each URL in cited_source_urls. "
            "Populated from ScoredSource.quality_score after the LLM call."
        ),
    )

    # Computed after LLM response — set before returning from verification.py
    evidence_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Locked formula (v3.1): "
            "avg(quality_score of cited sources) × min(1.0, cited_count / 3). "
            "Call compute_evidence_strength() to populate this field."
        ),
    )

    matched_claim_id: Optional[str] = Field(
        default=None,
        description="Set when this result was returned from claim matching cache (Step 3).",
    )
    pipeline_version: str = Field(default="v3.1")

    @model_validator(mode="after")
    def _validate_cited_lengths(self) -> "VerificationResult":
        if self.cited_source_urls and self.cited_source_qualities:
            if len(self.cited_source_urls) != len(self.cited_source_qualities):
                raise ValueError(
                    "cited_source_urls and cited_source_qualities must have the same length."
                )
        return self


def compute_evidence_strength(
    cited_source_qualities: list[float],
) -> float:
    """
    Locked formula (Pramana v3.1 Fix 16):

        evidence_strength = avg(quality_score of cited sources)
                            × min(1.0, cited_source_count / 3)

    Args:
        cited_source_qualities: quality_score for each source the LLM cited.
                                 Fetch from ScoredSource.quality_score post-LLM call.

    Returns:
        float in [0.0, 1.0].

    Examples:
        >>> compute_evidence_strength([])               # 0.0  — no sources cited
        >>> compute_evidence_strength([0.80])           # 0.267 — 1 source, weak
        >>> compute_evidence_strength([0.85, 0.85])     # 0.567 — 2 sources, moderate
        >>> compute_evidence_strength([0.80, 0.90, 0.85])  # 0.85 — 3 sources, strong
    """
    if not cited_source_qualities:
        return 0.0

    avg_quality = sum(cited_source_qualities) / len(cited_source_qualities)
    count_factor = min(1.0, len(cited_source_qualities) / 3.0)
    result = avg_quality * count_factor

    # Clamp to [0.0, 1.0] as a safeguard against floating-point edge cases.
    return round(max(0.0, min(1.0, result)), 4)


# ── Pipeline aggregate ─────────────────────────────────────────────────────────
class PipelineResult(BaseModel):
    """
    Top-level API response for a /verify call.
    Contains one VerificationResult per atomic claim extracted from the input.
    """

    input_text: str
    claims_found: int = Field(default=0)
    results: list[VerificationResult] = Field(default_factory=list)
    cached_count: int = Field(
        default=0,
        description="How many claims were served from the matching cache (Step 3).",
    )
    pipeline_version: str = Field(default="v3.1")


# Update forward references
ClaimMatchResult.model_rebuild()
