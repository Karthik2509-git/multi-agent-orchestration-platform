"""Mock LLM Provider for deterministic offline testing."""

from typing import Any, Callable, Dict, List, Optional

from src.app.llm.base import LLMProvider
from src.app.models.schemas.llm import LLMResponse

HandlerType = Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], LLMResponse]


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider that returns pre-queued responses or executes a handler."""

    def __init__(
        self,
        responses: Optional[List[LLMResponse]] = None,
        handler: Optional[HandlerType] = None,
        model: str = "mock-model",
    ):
        self.responses = list(responses) if responses else []
        self.handler = handler
        self.model = model
        self.history: List[Dict[str, Any]] = []

    def queue_response(self, response: LLMResponse) -> None:
        """Queue a response to be returned on the next generate() call."""
        self.responses.append(response)

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate a simulated response."""
        self.history.append({"messages": messages, "tools": tools})

        if self.handler:
            return self.handler(messages, tools)

        if self.responses:
            return self.responses.pop(0)

        # Default fallback response if nothing is queued
        last_message = messages[-1]["content"] if messages else ""
        return LLMResponse(
            content=f"Mock response to: {last_message}",
            tool_calls=[],
            model=self.model,
            finish_reason="stop",
        )
