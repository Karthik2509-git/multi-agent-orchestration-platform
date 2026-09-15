"""Health endpoint router."""

from fastapi import APIRouter, Depends, status

from src.app.core.config import Settings, get_settings
from src.app.models.schemas.health import HealthResponse
from src.app.services.health import get_health_status

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the current health and status of the application.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return application health information."""
    return get_health_status(settings)
