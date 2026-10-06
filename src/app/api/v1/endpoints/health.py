"""Health endpoint router."""

from fastapi import APIRouter, Depends, Response, status

from src.app.core.config import Settings, get_settings
from src.app.models.schemas.health import HealthResponse
from src.app.services.health import get_readiness_status

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "Required runtime dependencies unavailable",
        }
    },
    summary="Health Check",
    description="Check the current health and readiness status of the application.",
)
async def health_check(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return application readiness information."""
    is_ready, health_info = await get_readiness_status(settings)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health_info
