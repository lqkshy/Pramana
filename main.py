"""Pramana API — FastAPI fact-checking application.

Startup logs DEV_MODE from environment. Routes are included from
app/api/routes/verify.py at the root level.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()  # load .env from project root

DEV_MODE = os.getenv("DEV_MODE", "false").strip().lower() == "true"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Pramana API",
        version="0.1.0",
        # NOTE: DEV_MODE logged on startup below
    )

    # CORS — allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the verification router from the claims pipeline
    from app.api.routes.verify import router as verify_router
    app.include_router(verify_router, prefix="")

    return app


app = create_app()


@app.on_event("startup")
async def on_startup():
    """Log startup information including DEV_MODE."""
    import logging

    logger = logging.getLogger("uvicorn.error")
    logger.info(f"Pramana API starting. DEV_MODE={DEV_MODE}")