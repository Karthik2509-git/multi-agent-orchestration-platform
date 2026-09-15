"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.app.models.schemas.llm import LLMResponse


class LLMProvider(ABC):
    """Abstract interface defining the minimal contract for an LLM provider."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate a response or tool call decisions from the LLM.

        Args:
            messages: Conversation message history in standardized format.
            tools: Optional list of tool definitions formatted as JSON Schema functions.

        Returns:
            LLMResponse containing text output and/or tool calls.
        """
        pass
