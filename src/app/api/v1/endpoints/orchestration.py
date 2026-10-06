"""Multi-agent orchestration API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.models.schemas.orchestration import (
    OrchestrationRunRequest,
    OrchestrationRunResponse,
)
from src.app.replay.models import ReplayRequest, ReplayResult
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
            thread_id=request.thread_id,
            scope_id=request.scope_id,
            require_human_review=request.require_human_review,
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


@router.post(
    "/replay",
    response_model=ReplayResult,
    status_code=status.HTTP_200_OK,
    description=(
        "Developer execution fork / replay endpoint creating a new execution fork "
        "from an existing checkpoint with explicitly applied modifications."
    ),
)
async def replay_orchestration(
    request: ReplayRequest,
    settings: Settings = Depends(get_settings),
) -> ReplayResult:
    """Fork and replay an existing execution with validated modifications."""
    from src.app.replay.service import get_replay_service

    logger.info("Received execution replay request for thread '%s'", request.source_thread_id)
    service = get_replay_service(settings)
    result = await service.fork_and_replay(
        source_thread_id=request.source_thread_id,
        modifications=request.modifications,
    )
    if not result.success and result.error_category == "source_not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.error or "Source execution thread not found.",
        )
    return result
