"""Domain schemas for LLM responses and tool calling contracts."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool call request produced by the LLM."""

    id: str = Field(description="Unique ID of the tool call")
    name: str = Field(description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed argument parameters for the tool"
    )


class LLMResponse(BaseModel):
    """Standardized response schema returned by any LLMProvider."""

    content: Optional[str] = Field(default=None, description="Direct text answer from the model")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="List of tool calls requested by the model"
    )
    model: str = Field(default="", description="Identifier of the model that generated the output")
    finish_reason: str = Field(
        default="stop", description="Reason model stopped (e.g. stop, tool_calls)"
    )
