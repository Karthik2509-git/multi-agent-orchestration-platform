"""Base classes and contracts for agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Encapsulates the outcome of a tool execution."""

    success: bool = Field(description="Whether the tool execution succeeded")
    data: Any = Field(default=None, description="Output data produced by the tool")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")


class BaseTool(ABC):
    """Abstract base class for all agent-callable tools."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool logic with validated arguments."""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Format the tool definition into OpenAI function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
