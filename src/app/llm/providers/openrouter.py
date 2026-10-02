"""OpenRouter LLM Provider implementation using OpenAI-compatible API."""

import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAIError

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall

logger = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "https://github.com/Karthik2509-git/multi-agent-orchestration-platform"
OPENROUTER_APP_TITLE = "Multi-Agent Orchestration Platform"


class OpenRouterLLMProvider(LLMProvider):
    """LLM provider implementation for OpenRouter models."""

    def __init__(self, api_key: str, model: str = "openrouter/free"):
        if not api_key:
            raise ValueError(
                "OpenRouter API key must be provided to initialize OpenRouterLLMProvider"
            )
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_APP_TITLE,
            },
        )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call OpenRouter chat completions API with tools if provided."""
        logger.info(
            "Invoking OpenRouter model %s with %d messages and %d tools",
            self.model,
            len(messages),
            len(tools) if tools else 0,
        )

        import time

        from src.app.observability import record_span_error, set_span_attributes, trace_span

        formatted_tools = None
        if tools:
            formatted_tools = [{"type": "function", "function": tool} for tool in tools]

        attrs = {
            "provider": "openrouter",
            "model": self.model,
            "operation": "generate",
        }

        async with trace_span("llm.call", attributes=attrs) as span:
            start_time = time.perf_counter()
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                }
                if formatted_tools:
                    kwargs["tools"] = formatted_tools

                response = await self.client.chat.completions.create(**kwargs)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                choice = response.choices[0]

                parsed_tool_calls: List[ToolCall] = []
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        try:
                            args = (
                                json.loads(tc.function.arguments)
                                if isinstance(tc.function.arguments, str)
                                else (tc.function.arguments or {})
                            )
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to decode JSON arguments for tool %s",
                                tc.function.name,
                            )
                            args = {}
                        parsed_tool_calls.append(
                            ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                        )

                span_updates: Dict[str, Any] = {
                    "status": "success",
                    "latency_ms": latency_ms,
                    "model": response.model or self.model,
                }
                if hasattr(response, "usage") and response.usage:
                    if (
                        hasattr(response.usage, "prompt_tokens")
                        and response.usage.prompt_tokens is not None
                    ):
                        span_updates["llm.prompt_tokens"] = response.usage.prompt_tokens
                    if (
                        hasattr(response.usage, "completion_tokens")
                        and response.usage.completion_tokens is not None
                    ):
                        span_updates["llm.completion_tokens"] = response.usage.completion_tokens
                    if (
                        hasattr(response.usage, "total_tokens")
                        and response.usage.total_tokens is not None
                    ):
                        span_updates["llm.total_tokens"] = response.usage.total_tokens

                set_span_attributes(span, span_updates)

                return LLMResponse(
                    content=choice.message.content,
                    tool_calls=parsed_tool_calls,
                    model=response.model or self.model,
                    finish_reason=choice.finish_reason or "stop",
                )
            except OpenAIError as e:
                set_span_attributes(span, {"status": "error"})
                record_span_error(span, e)
                logger.error("OpenRouter API invocation failed: %s", type(e).__name__)
                raise RuntimeError(f"OpenRouter service error: {type(e).__name__}") from e
