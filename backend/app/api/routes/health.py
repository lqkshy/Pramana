"""
Health and readiness endpoints for Pramana.

Used by Docker / Kubernetes probes to determine whether the
service is alive and ready to serve traffic.
"""
from fastapi import APIRouter

from backend.app.core.config import get_settings

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe — indicates the service process is running."""
    return {"status": "healthy", "service": get_settings().app_name}


@router.get("/readiness")
async def readiness() -> dict:
    """Readiness probe — indicates the service is ready to serve traffic."""
    return {"status": "ready", "service": get_settings().app_name}
