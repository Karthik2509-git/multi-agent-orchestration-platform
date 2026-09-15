"""Request and response schemas for agent execution."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCallSummary(BaseModel):
    """Summary record of a tool invocation during agent execution."""

    tool_name: str = Field(description="Name of the invoked tool")
    arguments: Dict[str, Any] = Field(description="Arguments passed to the tool")
    result: Any = Field(default=None, description="Output returned by the tool")
    success: bool = Field(description="Whether the tool execution succeeded")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    duration_ms: float = Field(description="Execution time in milliseconds")


class AgentRunRequest(BaseModel):
    """Request payload for running an agent task."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The task or question to be processed by the agent",
        examples=["Calculate (1000 / 8) + 42", "Fetch the content of https://httpbin.org/get"],
    )


class AgentRunResponse(BaseModel):
    """Structured response returned by the agent API."""

    task: str = Field(description="The original user task")
    answer: str = Field(description="Final answer or synthesis produced by the agent")
    tool_calls: List[ToolCallSummary] = Field(
        default_factory=list, description="Audit trace of all tool calls executed"
    )
    model: str = Field(description="The underlying LLM model identifier")
    status: str = Field(
        description="Execution outcome: 'completed', 'max_iterations_reached', or 'error'"
    )
    execution_time_seconds: float = Field(description="Total agent execution time in seconds")
