"""
Health and readiness endpoints for Pramana.

Used by Docker / Kubernetes probes to determine whether the
service is alive and ready to serve traffic.
"""
from fastapi import APIRouter

router = APIRouter(tags=["ops"])

# TODO: implement

@router.get("/health")
async def health() -> dict:
    """Liveness probe."""
    pass

@router.get("/readiness")
async def readiness() -> dict:
    """Readiness probe — checks downstream dependencies."""
    pass
