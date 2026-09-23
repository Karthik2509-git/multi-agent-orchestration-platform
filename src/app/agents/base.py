"""Base contracts and abstractions for specialized worker agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.app.orchestration.state import OrchestrationState


class AgentResult(BaseModel):
    """Structured result returned by any specialized worker agent."""

    agent: str = Field(description="Identifier of the specialized agent")
    status: str = Field(default="success", description="Outcome status: 'success' or 'error'")
    result: str = Field(description="Detailed text result or synthesis from the agent")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional diagnostic metadata"
    )


class BaseSpecializedAgent(ABC):
    """Abstract base class for all specialized domain agents in the orchestration graph."""

    name: str

    @abstractmethod
    async def run(self, task: str, state: OrchestrationState) -> AgentResult:
        """Execute the agent's specialized task given current accumulated state."""
        pass
