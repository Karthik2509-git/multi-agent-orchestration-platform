"""Agent API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.models.schemas.agent import AgentRunRequest, AgentRunResponse
from src.app.services.agent_service import run_agent_task

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run AI Agent",
    description="Execute an autonomous task with dynamic tool calling (Calculator, Safe HTTP GET).",
)
async def run_agent(
    request: AgentRunRequest,
    settings: Settings = Depends(get_settings),
) -> AgentRunResponse:
    """Execute a task using the ToolCallingAgent."""
    logger.info("Received agent run request: '%s'", request.task[:80])

    try:
        response = await run_agent_task(
            task=request.task,
            settings=settings,
        )
        return response
    except ValueError as val_err:
        logger.warning("Configuration or validation error during agent run: %s", str(val_err))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Internal failure during agent run: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution encountered an internal error: {str(exc)}",
        )
