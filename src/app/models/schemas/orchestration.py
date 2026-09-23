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


class OrchestrationRunResponse(BaseModel):
    """Response payload returned by the multi-agent orchestration workflow."""

    task: str = Field(description="The original user task")
    answer: str = Field(description="The final synthesized answer")
    agents_used: List[str] = Field(
        default_factory=list,
        description="List of agent names that participated in completing the task",
    )
    status: str = Field(
        default="completed",
        description="Execution status ('completed' or 'error')",
    )
    execution_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed execution time in seconds",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional non-sensitive workflow metadata",
    )
