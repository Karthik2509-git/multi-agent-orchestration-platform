"""Multi-agent orchestration API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.models.schemas.orchestration import (
    OrchestrationRunRequest,
    OrchestrationRunResponse,
)
from src.app.services.orchestration_service import run_orchestrated_task

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/run",
    response_model=OrchestrationRunResponse,
    status_code=status.HTTP_200_OK,
    description=(
        "Execute a task orchestrated across specialized agents (research, data, code) "
        "managed by a supervisor in LangGraph."
    ),
)
async def run_orchestration(
    request: OrchestrationRunRequest,
    settings: Settings = Depends(get_settings),
) -> OrchestrationRunResponse:
    """Execute a task using the multi-agent supervisor graph."""
    logger.info("Received orchestration request: '%s'", request.task[:80])

    try:
        response = await run_orchestrated_task(
            task=request.task,
            settings=settings,
        )
        return response
    except ValueError as val_err:
        logger.warning(
            "Configuration or validation error during orchestration: %s",
            str(val_err),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Internal error during orchestration run: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration encountered an internal error: {str(exc)}",
        )
