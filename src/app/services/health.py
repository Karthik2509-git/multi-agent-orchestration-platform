"""Health service providing shared health and readiness check logic."""

from typing import Tuple

from src.app.core.config import Settings
from src.app.models.schemas.health import HealthResponse


def get_health_status(settings: Settings) -> HealthResponse:
    """Generate system health check response based on application settings."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )


async def get_readiness_status(settings: Settings) -> Tuple[bool, HealthResponse]:
    """Check application readiness including required runtime dependencies."""
    is_ready = True

    if settings.checkpoint_backend == "postgres":
        from src.app.memory.working_memory import check_postgres_readiness

        is_ready = await check_postgres_readiness()

    status_str = "healthy" if is_ready else "unhealthy"

    response = HealthResponse(
        status=status_str,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )
    return is_ready, response
