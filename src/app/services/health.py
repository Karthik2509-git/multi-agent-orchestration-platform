"""Health service providing shared health check logic."""

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
