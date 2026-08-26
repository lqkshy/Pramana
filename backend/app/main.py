# FastAPI app factory that mounts API routers and configures logging
from fastapi import FastAPI

from backend.app.api.routes import claims, health
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    get_settings()
    app = FastAPI(title="Pramana", version="0.1.0")
    app.include_router(health.router)
    app.include_router(claims.router)
    return app


app = create_app()
