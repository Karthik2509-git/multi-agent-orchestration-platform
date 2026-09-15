"""FastAPI application initialization and lifespan management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.v1.api import api_v1_router
from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger, setup_logging
from src.app.models.schemas.health import HealthResponse
from src.app.services.health import get_health_status

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger.info(
        "Starting %s [version=%s, env=%s]",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure an instance of the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade Multi-Agent AI Orchestration Platform API",
        docs_url="/docs" if settings.debug or settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.debug or settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # Configure CORS with explicitly configured allowed origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root health endpoint (delegates to the shared health service)
    @app.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["Health"],
        summary="Root Health Check",
        description="Returns basic application health and runtime status.",
    )
    async def root_health(
        current_settings: Settings = Depends(get_settings),
    ) -> HealthResponse:
        return get_health_status(current_settings)

    # Mount versioned API routes
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


app = create_app()
