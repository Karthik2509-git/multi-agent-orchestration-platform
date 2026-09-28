"""REST API endpoints for Human-in-the-Loop (HITL) approval workflows."""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.hitl.models import HITLResponse
from src.app.hitl.service import HITLService
from src.app.llm.base import LLMProvider
from src.app.llm.factory import get_llm_provider
from src.app.memory.service import get_memory_service
from src.app.memory.working_memory import get_checkpointer
from src.app.models.schemas.hitl import (
    PendingApprovalResponse,
    ResumeTaskRequest,
    ResumeTaskResponse,
)
from src.app.orchestration.graph import build_orchestration_graph
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/pending/{thread_id}",
    response_model=PendingApprovalResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect Pending Approval",
    description="Inspect details and context of an interrupted workflow pending human review.",
)
async def get_pending_approval(
    thread_id: str,
    settings: Settings = Depends(get_settings),
) -> PendingApprovalResponse:
    checkpointer = await get_checkpointer(settings)
    hitl_service = HITLService(checkpointer=checkpointer, settings=settings)
    request = await hitl_service.get_pending_approval(thread_id)
    if not request:
        return PendingApprovalResponse(
            thread_id=thread_id,
            has_pending_approval=False,
            request=None,
        )

    return PendingApprovalResponse(
        thread_id=thread_id,
        has_pending_approval=True,
        request=request,
    )


@router.post(
    "/resume/{thread_id}",
    response_model=ResumeTaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Resume Interrupted Task",
    description=(
        "Submit a human decision (approve, reject, modify, take_over) "
        "to resume a paused execution thread."
    ),
)
async def resume_task(
    thread_id: str,
    request: ResumeTaskRequest,
    settings: Settings = Depends(get_settings),
) -> ResumeTaskResponse:
    logger.info(
        "Received resume request for thread '%s' (decision: '%s')",
        thread_id,
        request.decision.value,
    )

    provider: LLMProvider = get_llm_provider(settings)
    checkpointer = await get_checkpointer(settings)
    hitl_service = HITLService(checkpointer=checkpointer, settings=settings)
    memory_service = get_memory_service(settings) if settings.memory_enabled else None

    http_tool = SafeHTTPGetTool(
        allowed_domains=settings.allowed_http_domains,
        timeout=settings.tool_http_timeout,
        max_size_bytes=settings.tool_http_max_size_bytes,
    )
    calculator = CalculatorTool()

    graph = build_orchestration_graph(
        provider=provider,
        http_tool=http_tool,
        calculator=calculator,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )

    hitl_resp = HITLResponse(
        decision=request.decision,
        feedback=request.feedback,
        modified_action=request.modified_action,
        override_output=request.override_output,
    )

    start_time = time.perf_counter()
    try:
        final_state = await hitl_service.resume_thread(
            thread_id=thread_id,
            response=hitl_resp,
            graph=graph,
        )
        elapsed = round(time.perf_counter() - start_time, 3)

        # Check if the resumed execution was paused again by another interrupt
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        is_interrupted = False
        pending_payload = None

        if snapshot and getattr(snapshot, "tasks", None):
            for t in snapshot.tasks:
                if hasattr(t, "interrupts") and t.interrupts:
                    is_interrupted = True
                    first_int = t.interrupts[0]
                    pending_payload = first_int.value if hasattr(first_int, "value") else first_int
                    break

        if not is_interrupted and isinstance(final_state, dict) and "__interrupt__" in final_state:
            interrupts = final_state["__interrupt__"]
            if interrupts:
                is_interrupted = True
                first_int = interrupts[0]
                pending_payload = first_int.value if hasattr(first_int, "value") else first_int

        if is_interrupted:
            return ResumeTaskResponse(
                thread_id=thread_id,
                status="interrupted",
                answer="Execution paused awaiting subsequent human review.",
                agents_used=final_state.get("agents_used", [])
                if isinstance(final_state, dict)
                else [],
                execution_time_seconds=elapsed,
                pending_approval=pending_payload,
                metadata=final_state.get("metadata") if isinstance(final_state, dict) else None,
            )

        answer_text = (
            final_state.get("final_answer")
            if isinstance(final_state, dict) and final_state.get("final_answer")
            else "Task completed."
        )
        status_text = (
            final_state.get("status", "completed") if isinstance(final_state, dict) else "completed"
        )
        agents_used_list = (
            final_state.get("agents_used", []) if isinstance(final_state, dict) else []
        )

        return ResumeTaskResponse(
            thread_id=thread_id,
            status=status_text,
            answer=answer_text,
            agents_used=agents_used_list,
            execution_time_seconds=elapsed,
            metadata=final_state.get("metadata") if isinstance(final_state, dict) else None,
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - start_time, 3)
        logger.error("Failed to resume thread '%s': %s", thread_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume thread '{thread_id}': {str(e)}",
        )
