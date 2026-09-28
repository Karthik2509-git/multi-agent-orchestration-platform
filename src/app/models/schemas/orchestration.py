"""Schemas for multi-agent orchestration requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OrchestrationRunRequest(BaseModel):
    """Request payload for running multi-agent orchestration."""

    task: str = Field(
        ...,
        min_length=1,
        description="The task or goal to be orchestrated across specialized agents",
        examples=["Analyze the revenue growth of Apple from 2021 to 2023 and compute the CAGR"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional LangGraph thread identifier for stateful multi-turn runs",
    )
    scope_id: str = Field(
        default="default",
        description="Memory scope identifier for isolating user/task memories",
    )
    require_human_review: bool = Field(
        default=False,
        description="Explicitly require human approval before completing the task",
    )


class OrchestrationRunResponse(BaseModel):
    """Response payload returned by the multi-agent orchestration workflow."""

    task: str = Field(description="The original user task")
    answer: str = Field(description="The final synthesized answer or status description")
    thread_id: Optional[str] = Field(
        default=None,
        description="The LangGraph thread identifier associated with this run",
    )
    agents_used: List[str] = Field(
        default_factory=list,
        description="List of agent names that participated in completing the task",
    )
    status: str = Field(
        default="completed",
        description="Execution status ('completed', 'interrupted', or 'error')",
    )
    execution_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed execution time in seconds",
    )
    memories_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant long-term memories retrieved and utilized during planning",
    )
    pending_approval: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured approval request details if the task was interrupted",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional non-sensitive workflow metadata",
    )
