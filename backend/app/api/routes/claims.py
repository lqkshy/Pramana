"""
FastAPI route handlers for Pramana's claim-checking API.

Exposes POST /claims to submit text for fact-checking and
GET /claims/{claim_id} to poll for results.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/claims", tags=["claims"])

# TODO: implement

@router.post("/")
async def submit_claim(payload: dict) -> dict:  # replace dict with real schemas
    """Accept raw text and enqueue it for the fact-checking pipeline."""
    pass

@router.get("/{claim_id}")
async def get_claim_status(claim_id: str) -> dict:
    """Return the current verdict and evidence for a previously submitted claim."""
    pass
