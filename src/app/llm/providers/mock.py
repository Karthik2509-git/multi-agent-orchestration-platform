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
        import time

        from src.app.observability import (
            record_llm_metrics,
            record_span_error,
            set_span_attributes,
            trace_span,
        )

        attrs = {
            "provider": "mock",
            "model": self.model,
            "operation": "generate",
        }

        async with trace_span("llm.call", attributes=attrs) as span:
            start_time = time.perf_counter()
            self.history.append({"messages": messages, "tools": tools})

            try:
                if self.handler:
                    resp = self.handler(messages, tools)
                elif self.responses:
                    resp = self.responses.pop(0)
                else:
                    # Default fallback response if nothing is queued
                    last_message = messages[-1]["content"] if messages else ""
                    resp = LLMResponse(
                        content=f"Mock response to: {last_message}",
                        tool_calls=[],
                        model=self.model,
                        finish_reason="stop",
                    )

                latency_s = time.perf_counter() - start_time
                latency_ms = round(latency_s * 1000, 2)
                set_span_attributes(
                    span,
                    {
                        "status": "success",
                        "latency_ms": latency_ms,
                    },
                )
                prompt_tokens = getattr(resp, "prompt_tokens", getattr(self, "prompt_tokens", None))
                completion_tokens = getattr(
                    resp, "completion_tokens", getattr(self, "completion_tokens", None)
                )
                record_llm_metrics(
                    provider="mock",
                    model=self.model,
                    latency_seconds=latency_s,
                    status="success",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                return resp
            except Exception as e:
                latency_s = time.perf_counter() - start_time
                record_span_error(span, e)
                set_span_attributes(span, {"status": "error"})
                record_llm_metrics(
                    provider="mock",
                    model=self.model,
                    latency_seconds=latency_s,
                    status="error",
                )
                raise
