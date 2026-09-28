"""Human-in-the-Loop coordination service for inspecting and resuming interrupted threads."""

from typing import Any, Dict, Optional

from fastapi import Depends
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.hitl.models import ApprovalLevel, HITLRequest, HITLResponse
from src.app.hitl.policies import EscalationPolicy
from src.app.memory.working_memory import get_checkpointer

logger = get_logger(__name__)


class HITLService:
    """Coordinates inspection of paused threads and handles human decision resumption."""

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        policy: Optional[EscalationPolicy] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.checkpointer = checkpointer
        self.policy = policy or EscalationPolicy(
            confidence_threshold=self.settings.hitl_confidence_threshold,
            max_retries=self.settings.hitl_max_specialist_retries,
        )

    async def _get_saver(self) -> BaseCheckpointSaver:
        if self.checkpointer is None:
            self.checkpointer = await get_checkpointer(self.settings)
        return self.checkpointer

    async def get_pending_approval(self, thread_id: str) -> Optional[HITLRequest]:
        """Inspect the checkpointer state for a pending interrupt on the target thread."""
        saver = await self._get_saver()
        config = {"configurable": {"thread_id": thread_id}}

        state_tuple = await saver.aget_tuple(config)
        if state_tuple is None:
            return None

        # 1. Inspect pending_writes for __interrupt__
        if state_tuple.pending_writes:
            for _, channel, writes in state_tuple.pending_writes:
                if channel == "__interrupt__":
                    for item in writes:
                        interrupt_val = item.value if hasattr(item, "value") else item
                        if isinstance(interrupt_val, dict):
                            raw_level = interrupt_val.get(
                                "level", ApprovalLevel.APPROVE_ACTION.value
                            )
                            try:
                                level = ApprovalLevel(raw_level)
                            except ValueError:
                                level = ApprovalLevel.APPROVE_ACTION
                            return HITLRequest(
                                interrupt_id=str(interrupt_val.get("interrupt_id", "unknown")),
                                thread_id=thread_id,
                                action_type=interrupt_val.get("action_type", "human_review"),
                                action_details=interrupt_val.get("action_details", {}),
                                reason=interrupt_val.get(
                                    "reason", "Action paused awaiting human review."
                                ),
                                context_summary=interrupt_val.get("context_summary", ""),
                                level=level,
                            )

        # 2. Also inspect metadata in checkpoint values if stored
        state_values = state_tuple.checkpoint.get("channel_values", {})
        metadata = state_values.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("pending_hitl"):
            hitl_dict = metadata["pending_hitl"]
            return HITLRequest(**hitl_dict)

        return None

    async def resume_thread(
        self,
        thread_id: str,
        response: HITLResponse,
        graph: Any,
    ) -> Dict[str, Any]:
        """Resume execution of an interrupted thread using LangGraph Command(resume=...)."""
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(
            "Resuming thread '%s' with human decision '%s'",
            thread_id,
            response.decision.value,
        )

        # Handle TAKE_OVER override decision at service level if needed
        resume_payload: Dict[str, Any] = {
            "decision": response.decision.value,
            "feedback": response.feedback,
            "modified_action": response.modified_action,
            "override_output": response.override_output,
        }

        # Submit Command to resume graph execution
        command = Command(resume=resume_payload)
        final_state = await graph.ainvoke(command, config=config)
        return final_state


_global_hitl_service: Optional[HITLService] = None


async def get_hitl_service(
    settings: Optional[Settings] = Depends(get_settings),
) -> HITLService:
    """Dependency provider returning singleton HITLService."""
    global _global_hitl_service
    if _global_hitl_service is None:
        app_settings = settings or get_settings()
        checkpointer = await get_checkpointer(app_settings)
        _global_hitl_service = HITLService(checkpointer=checkpointer, settings=app_settings)
    return _global_hitl_service


def reset_hitl_service() -> None:
    """Reset singleton HITL service instance."""
    global _global_hitl_service
    _global_hitl_service = None
