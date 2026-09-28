"""Service coordinating multi-agent orchestration execution with Memory and HITL."""

import time
from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.app.core.config import Settings
from src.app.core.logging import get_logger
from src.app.hitl.policies import EscalationPolicy
from src.app.llm.base import LLMProvider
from src.app.llm.factory import get_llm_provider
from src.app.memory.service import MemoryService, get_memory_service
from src.app.memory.working_memory import get_checkpointer
from src.app.models.schemas.orchestration import OrchestrationRunResponse
from src.app.orchestration.graph import build_orchestration_graph
from src.app.orchestration.state import OrchestrationState
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool

logger = get_logger(__name__)


async def run_orchestrated_task(
    task: str,
    settings: Settings,
    thread_id: Optional[str] = None,
    scope_id: str = "default",
    require_human_review: bool = False,
    provider: Optional[LLMProvider] = None,
    http_tool: Optional[SafeHTTPGetTool] = None,
    calculator: Optional[CalculatorTool] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    memory_service: Optional[MemoryService] = None,
    hitl_policy: Optional[EscalationPolicy] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> OrchestrationRunResponse:
    """Coordinate end-to-end multi-agent orchestration for a given task with Memory and HITL."""
    active_thread_id = thread_id or str(uuid4())

    if provider is None:
        provider = get_llm_provider(settings)

    if http_tool is None:
        http_tool = SafeHTTPGetTool(
            allowed_domains=settings.allowed_http_domains,
            timeout=settings.tool_http_timeout,
            max_size_bytes=settings.tool_http_max_size_bytes,
        )

    if calculator is None:
        calculator = CalculatorTool()

    if checkpointer is None:
        checkpointer = await get_checkpointer(settings)

    if memory_service is None and settings.memory_enabled:
        memory_service = get_memory_service(settings)

    graph = build_orchestration_graph(
        provider=provider,
        http_tool=http_tool,
        calculator=calculator,
        checkpointer=checkpointer,
        memory_service=memory_service,
        hitl_policy=hitl_policy,
    )

    metadata: Dict[str, Any] = {
        "scope_id": scope_id,
        "require_human_review": require_human_review,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    initial_state: OrchestrationState = {
        "task": task,
        "messages": [{"role": "user", "content": task}],
        "next_agent": "",
        "agent_results": {},
        "agents_used": [],
        "step_count": 0,
        "final_answer": "",
        "status": "pending",
        "metadata": metadata,
        "memories_used": [],
    }

    config = {"configurable": {"thread_id": active_thread_id}}
    start_time = time.perf_counter()
    logger.info(
        "Starting multi-agent orchestration for thread '%s' (task: '%s')",
        active_thread_id,
        task[:80],
    )

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        execution_time = round(time.perf_counter() - start_time, 3)

        # Inspect if execution was paused by an interrupt
        snapshot = await graph.aget_state(config)
        is_interrupted = False
        pending_approval_payload = None

        if snapshot and getattr(snapshot, "tasks", None):
            for t in snapshot.tasks:
                if hasattr(t, "interrupts") and t.interrupts:
                    is_interrupted = True
                    first_int = t.interrupts[0]
                    pending_approval_payload = (
                        first_int.value if hasattr(first_int, "value") else first_int
                    )
                    break

        if not is_interrupted and isinstance(final_state, dict) and "__interrupt__" in final_state:
            interrupts = final_state["__interrupt__"]
            if interrupts:
                is_interrupted = True
                first_int = interrupts[0]
                pending_approval_payload = (
                    first_int.value if hasattr(first_int, "value") else first_int
                )

        if is_interrupted:
            logger.info("Workflow paused on thread '%s' awaiting human review.", active_thread_id)
            return OrchestrationRunResponse(
                task=task,
                answer="Execution paused awaiting human review.",
                thread_id=active_thread_id,
                agents_used=(
                    final_state.get("agents_used", []) if isinstance(final_state, dict) else []
                ),
                status="interrupted",
                execution_time_seconds=execution_time,
                memories_used=(
                    final_state.get("memories_used", []) if isinstance(final_state, dict) else []
                ),
                pending_approval=pending_approval_payload,
                metadata=metadata,
            )

        answer_text = (
            final_state.get("final_answer")
            if isinstance(final_state, dict) and final_state.get("final_answer")
            else "No answer produced."
        )
        status_text = (
            final_state.get("status", "completed") if isinstance(final_state, dict) else "completed"
        )
        agents_used_list = (
            final_state.get("agents_used", []) if isinstance(final_state, dict) else []
        )
        memories_used_list = (
            final_state.get("memories_used", []) if isinstance(final_state, dict) else []
        )

        logger.info(
            "Multi-agent orchestration completed in %.2fs with status '%s'. Agents used: %s",
            execution_time,
            status_text,
            agents_used_list,
        )

        return OrchestrationRunResponse(
            task=task,
            answer=answer_text,
            thread_id=active_thread_id,
            agents_used=agents_used_list,
            status=status_text,
            execution_time_seconds=execution_time,
            memories_used=memories_used_list,
            metadata=final_state.get("metadata") if isinstance(final_state, dict) else metadata,
        )
    except Exception as e:
        execution_time = round(time.perf_counter() - start_time, 3)
        logger.error("Multi-agent orchestration failed after %.2fs: %s", execution_time, str(e))
        return OrchestrationRunResponse(
            task=task,
            answer=f"Orchestration failed: {str(e)}",
            thread_id=active_thread_id,
            agents_used=initial_state.get("agents_used", []),
            status="error",
            execution_time_seconds=execution_time,
            metadata={"error": str(e)},
        )
