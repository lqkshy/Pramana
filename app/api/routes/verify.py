"""API routes for claim verification.

Exports an APIRouter with:
  GET /health  — health check with provider/dev_mode info
  POST /verify — verify claims extracted from input text
"""

from fastapi import APIRouter, Body
from typing import List

from app.models.schemas import (
    ClaimInput,
    VerificationResult,
    VerifyResponse,
)
from app.pipeline.claims.extractor import extract_claims


router = APIRouter(prefix="")


@router.get("/health")
async def health_check():
    """Return health status including provider and dev_mode."""
    from os import getenv
    from dotenv import load_dotenv

    load_dotenv()  # ensure env vars are loaded
    provider = getenv("LLM_PROVIDER", "groq")
    dev_mode = getenv("DEV_MODE", "false").strip().lower() == "true"

    return {
        "status": "ok",
        "provider": provider,
        "dev_mode": dev_mode,
    }


@router.post("/verify", response_model=VerifyResponse)
async def verify_claims(
    claim_input: ClaimInput = Body(...),
):
    """Extract claims from text and return mock verification results.

    For each selected claim, a VerificationResult is created with
    verdict="SUPPORTED", confidence=0.5, evidence_strength=0.5,
    and a placeholder explanation.
    """
    # Extract claims from the input text
    extracted = await extract_claims(claim_input.text)

    # Build a VerificationResult for each selected claim
    results: List[VerificationResult] = []
    for claim in extracted.get("selected_claims", []):
        results.append(
            VerificationResult(
                claim=claim,
                verdict="SUPPORTED",  # mocked for now
                confidence=0.5,
                evidence_strength=0.5,
                explanation="Full verification coming in Week 5",
                matched_claim_id=None,
            )
        )

    # Ensure at least one claim result per the VerifyResponse contract
    if not results:
        results.append(
            VerificationResult(
                claim="[no claims extracted]",
                verdict="SUPPORTED",
                confidence=0.5,
                evidence_strength=0.5,
                explanation="Full verification coming in Week 5",
                matched_claim_id=None,
            )
        )

    return VerifyResponse(claims=results)