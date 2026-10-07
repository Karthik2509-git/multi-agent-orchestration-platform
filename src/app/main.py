"""FastAPI application initialization and lifespan management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.app.api.v1.api import api_v1_router
from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger, setup_logging
from src.app.memory.working_memory import close_checkpointer, init_checkpointer
from src.app.models.schemas.health import HealthResponse
from src.app.observability import init_telemetry, shutdown_telemetry
from src.app.services.health import get_health_status
from src.app.services.mcp_service import get_mcp_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events."""
    get_settings_fn = app.dependency_overrides.get(get_settings, get_settings)
    settings = get_settings_fn()
    setup_logging(log_level=settings.log_level)
    logger.info(
        "Starting %s [version=%s, env=%s]",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )

    # Initialize telemetry subsystem
    init_telemetry(settings)

    # Initialize checkpointer pool if configured
    try:
        await init_checkpointer(settings)
    except Exception as e:
        logger.warning("Checkpointer initialization deferred or failed: %s", e)

    # Initialize MCP subsystem if enabled
    mcp_service = get_mcp_service(settings)
    if settings.mcp_enabled:
        await mcp_service.initialize()

    yield

    # Teardown MCP subsystem if enabled
    if settings.mcp_enabled:
        await mcp_service.shutdown()

    # Teardown checkpointer pool
    await close_checkpointer()

    # Teardown telemetry subsystem
    shutdown_telemetry()

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure an instance of the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade Multi-Agent AI Orchestration Platform API",
        docs_url="/docs"
        if settings.debug or settings.app_env != "production" or settings.public_docs_enabled
        else None,
        redoc_url="/redoc"
        if settings.debug or settings.app_env != "production" or settings.public_docs_enabled
        else None,
        lifespan=lifespan,
    )

    # Instrument with OpenTelemetry if enabled
    if settings.telemetry_enabled:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="docs,redoc,openapi.json",
        )

    # Configure CORS with explicitly configured allowed origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root() -> str:
        """Return a small portfolio-friendly landing page for the API."""
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Agent AI Orchestration Platform</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 760px;
      margin: 4rem auto;
      padding: 0 1.25rem;
      line-height: 1.6;
    }
    a { margin-right: 1rem; }
    code { background: #f3f4f6; padding: .15rem .35rem; border-radius: .25rem; }
  </style>
</head>
<body>
  <h1>Multi-Agent AI Orchestration Platform</h1>
  <p>
    A FastAPI backend for tool-using, memory-enabled multi-agent workflows with
    LangGraph, MCP, RAG, HITL, observability, and execution replay.
  </p>
  <p>
    <a href="/docs">Swagger UI</a>
    <a href="/redoc">ReDoc</a>
    <a href="/openapi.json">OpenAPI JSON</a>
    <a href="/health">Health</a>
  </p>
  <p>API base: <code>/api/v1</code></p>
</body>
</html>"""

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
