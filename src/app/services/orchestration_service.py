"""Service coordinating multi-agent orchestration execution."""

import time
from typing import Optional

from src.app.core.config import Settings
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.llm.factory import get_llm_provider
from src.app.models.schemas.orchestration import OrchestrationRunResponse
from src.app.orchestration.graph import build_orchestration_graph
from src.app.orchestration.state import OrchestrationState
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool

logger = get_logger(__name__)


async def run_orchestrated_task(
    task: str,
    settings: Settings,
    provider: Optional[LLMProvider] = None,
    http_tool: Optional[SafeHTTPGetTool] = None,
    calculator: Optional[CalculatorTool] = None,
) -> OrchestrationRunResponse:
    """Coordinate end-to-end multi-agent orchestration for a given task."""
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

    graph = build_orchestration_graph(
        provider=provider,
        http_tool=http_tool,
        calculator=calculator,
    )

    initial_state: OrchestrationState = {
        "task": task,
        "messages": [{"role": "user", "content": task}],
        "next_agent": "",
        "agent_results": {},
        "agents_used": [],
        "step_count": 0,
        "final_answer": "",
        "status": "pending",
        "metadata": {},
    }

    start_time = time.perf_counter()
    logger.info("Starting multi-agent orchestration workflow for task: '%s'", task[:80])

    try:
        final_state = await graph.ainvoke(initial_state)
        execution_time = round(time.perf_counter() - start_time, 3)

        logger.info(
            "Multi-agent orchestration completed in %.2fs with status '%s'. Agents used: %s",
            execution_time,
            final_state.get("status", "completed"),
            final_state.get("agents_used", []),
        )

        return OrchestrationRunResponse(
            task=task,
            answer=final_state.get("final_answer") or "No answer produced.",
            agents_used=final_state.get("agents_used", []),
            status=final_state.get("status", "completed"),
            execution_time_seconds=execution_time,
            metadata=final_state.get("metadata"),
        )
    except Exception as e:
        execution_time = round(time.perf_counter() - start_time, 3)
        logger.error("Multi-agent orchestration failed after %.2fs: %s", execution_time, str(e))
        return OrchestrationRunResponse(
            task=task,
            answer=f"Orchestration failed: {str(e)}",
            agents_used=initial_state.get("agents_used", []),
            status="error",
            execution_time_seconds=execution_time,
            metadata={"error": str(e)},
        )
