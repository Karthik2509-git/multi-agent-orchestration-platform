"""Developer Execution Fork / Replay service."""

import time
from typing import Optional
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.hitl.policies import EscalationPolicy
from src.app.llm.base import LLMProvider
from src.app.llm.factory import get_llm_provider
from src.app.memory.service import MemoryService
from src.app.memory.working_memory import get_checkpointer
from src.app.observability import record_span_error, set_span_attributes, trace_span
from src.app.replay.models import ReplayModification, ReplayResult
from src.app.services.orchestration_service import run_orchestrated_task
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool
from src.app.tools.registry import ToolRegistry

logger = get_logger(__name__)


class ReplayService:
    """Service orchestrating execution forks and replays from existing checkpoints."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.settings = settings or get_settings()
        self.checkpointer = checkpointer
        self.tool_registry = tool_registry

    async def _get_saver(self) -> BaseCheckpointSaver:
        if self.checkpointer is None:
            self.checkpointer = await get_checkpointer(self.settings)
        return self.checkpointer

    async def fork_and_replay(
        self,
        source_thread_id: str,
        modifications: Optional[ReplayModification] = None,
        provider: Optional[LLMProvider] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        tool_registry: Optional[ToolRegistry] = None,
        http_tool: Optional[SafeHTTPGetTool] = None,
        calculator: Optional[CalculatorTool] = None,
        memory_service: Optional[MemoryService] = None,
        hitl_policy: Optional[EscalationPolicy] = None,
    ) -> ReplayResult:
        """Create a new execution fork from a prior checkpoint with explicitly applied
        modifications.

        Safety boundaries:
        - The source thread checkpoint is never mutated or overwritten.
        - A unique new thread ID is generated for the fork.
        - Modifications are strictly validated through ReplayModification models.
        - Tool budgets, failure isolation, and guardrails apply to the forked execution.
        """
        start_time = time.perf_counter()
        saver = checkpointer or await self._get_saver()

        # 1. Verify existence of source execution in checkpointer
        source_config = {"configurable": {"thread_id": source_thread_id}}
        state_tuple = await saver.aget_tuple(source_config)
        if state_tuple is None or not state_tuple.checkpoint:
            logger.warning("Replay requested for non-existent thread '%s'", source_thread_id)
            return ReplayResult(
                success=False,
                source_thread_id=source_thread_id,
                status="error",
                error=f"Source execution thread '{source_thread_id}' not found.",
                error_category="source_not_found",
            )

        channel_values = state_tuple.checkpoint.get("channel_values", {})
        source_task = str(channel_values.get("task", ""))
        source_metadata = dict(channel_values.get("metadata", {}))

        # 2. Derive modified parameters
        effective_task = (
            modifications.task_override
            if (modifications and modifications.task_override)
            else source_task
        )
        if not effective_task:
            effective_task = "Replay execution task"

        effective_scope = (
            modifications.scope_id_override
            if (modifications and modifications.scope_id_override)
            else str(source_metadata.get("scope_id", "default"))
        )

        # 3. Model / provider resolution
        effective_provider = provider
        if modifications and modifications.model_override:
            override = modifications.model_override
            provider_kwargs = {}
            if override.model:
                provider_kwargs[f"{override.provider}_model"] = override.model
            temp_settings = self.settings.model_copy(
                update={"llm_provider": override.provider, **provider_kwargs}
            )
            try:
                effective_provider = get_llm_provider(temp_settings)
            except Exception as prov_err:
                logger.warning("Model override rejected: %s", prov_err)
                return ReplayResult(
                    success=False,
                    source_thread_id=source_thread_id,
                    status="error",
                    error=f"Model override rejected: {prov_err}",
                    error_category="validation_error",
                )

        # 4. Tool Registry / Mock Tool resolution
        base_registry = tool_registry or self.tool_registry or ToolRegistry()
        mock_tools = modifications.mock_tool_results if modifications else None
        effective_registry = base_registry.fork(mock_tool_results=mock_tools)

        # 5. Generate NEW thread identity
        fork_thread_id = str(uuid4())

        applied_mods_dict = modifications.model_dump(exclude_none=True) if modifications else {}

        fork_metadata = {
            **source_metadata,
            "scope_id": effective_scope,
            "is_replay": True,
            "source_thread_id": source_thread_id,
            "replay_thread_id": fork_thread_id,
            "applied_modifications": applied_mods_dict,
        }

        if modifications and modifications.supervisor_route_override:
            fork_metadata["supervisor_route_override"] = modifications.supervisor_route_override

        # 6. Observability: low-cardinality span attributes (no query text or thread IDs)
        span_attrs = {
            "execution_type": "replay",
            "has_task_override": bool(modifications and modifications.task_override),
            "has_model_override": bool(modifications and modifications.model_override),
            "has_route_override": bool(modifications and modifications.supervisor_route_override),
            "has_mock_tool_results": bool(modifications and modifications.mock_tool_results),
            "has_scope_override": bool(modifications and modifications.scope_id_override),
        }

        async with trace_span("replay.run", attributes=span_attrs) as span:
            try:
                run_response = await run_orchestrated_task(
                    task=effective_task,
                    settings=self.settings,
                    thread_id=fork_thread_id,
                    scope_id=effective_scope,
                    require_human_review=source_metadata.get("require_human_review", False),
                    provider=effective_provider,
                    http_tool=http_tool,
                    calculator=calculator,
                    checkpointer=saver,
                    memory_service=memory_service,
                    hitl_policy=hitl_policy,
                    extra_metadata=fork_metadata,
                    tool_registry=effective_registry,
                )

                elapsed = round(time.perf_counter() - start_time, 3)
                set_span_attributes(span, {"status": run_response.status})

                return ReplayResult(
                    success=(run_response.status in ("completed", "interrupted")),
                    source_thread_id=source_thread_id,
                    replay_thread_id=fork_thread_id,
                    status=run_response.status,
                    applied_modifications=applied_mods_dict,
                    answer=run_response.answer,
                    agents_used=run_response.agents_used,
                    execution_time_seconds=elapsed,
                    is_interrupted=(run_response.status == "interrupted"),
                    pending_approval=run_response.pending_approval,
                    error=None if run_response.status != "error" else run_response.answer,
                    error_category=None if run_response.status != "error" else "execution_error",
                    metadata=run_response.metadata or fork_metadata,
                )
            except Exception as e:
                elapsed = round(time.perf_counter() - start_time, 3)
                logger.error("Replay execution failed: %s", e)
                record_span_error(span, e)
                return ReplayResult(
                    success=False,
                    source_thread_id=source_thread_id,
                    replay_thread_id=fork_thread_id,
                    status="error",
                    applied_modifications=applied_mods_dict,
                    execution_time_seconds=elapsed,
                    error=str(e),
                    error_category="execution_error",
                    metadata=fork_metadata,
                )


_replay_service_instance: Optional[ReplayService] = None


def get_replay_service(settings: Optional[Settings] = None) -> ReplayService:
    """Return singleton ReplayService instance."""
    global _replay_service_instance
    if _replay_service_instance is None:
        _replay_service_instance = ReplayService(settings=settings)
    return _replay_service_instance
