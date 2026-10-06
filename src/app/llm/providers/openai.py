"""OpenAI LLM Provider implementation using official AsyncOpenAI SDK."""

import json
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAIError

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.observability import (
    record_llm_metrics,
    record_span_error,
    set_span_attributes,
    trace_span,
)

logger = get_logger(__name__)


class OpenAILLMProvider(LLMProvider):
    """LLM provider implementation for OpenAI models."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API key must be provided to initialize OpenAILLMProvider")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Call OpenAI chat completions API with tools if provided."""
        logger.info(
            "Invoking OpenAI model %s with %d messages and %d tools",
            self.model,
            len(messages),
            len(tools) if tools else 0,
        )

        formatted_tools = None
        if tools:
            formatted_tools = [{"type": "function", "function": tool} for tool in tools]

        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        attrs = {
            "provider": "openai",
            "prompt_messages_count": len(messages),
            "prompt_chars": prompt_chars,
            "has_tools": bool(tools),
            "tools_count": len(tools) if tools else 0,
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
                latency_s = time.perf_counter() - start_time
                latency_ms = round(latency_s * 1000, 2)
                choice = response.choices[0]

                parsed_tool_calls: List[ToolCall] = []
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
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
                prompt_tokens = None
                completion_tokens = None
                if hasattr(response, "usage") and response.usage:
                    if (
                        hasattr(response.usage, "prompt_tokens")
                        and response.usage.prompt_tokens is not None
                    ):
                        prompt_tokens = response.usage.prompt_tokens
                        span_updates["llm.prompt_tokens"] = prompt_tokens
                    if (
                        hasattr(response.usage, "completion_tokens")
                        and response.usage.completion_tokens is not None
                    ):
                        completion_tokens = response.usage.completion_tokens
                        span_updates["llm.completion_tokens"] = completion_tokens
                    if (
                        hasattr(response.usage, "total_tokens")
                        and response.usage.total_tokens is not None
                    ):
                        span_updates["llm.total_tokens"] = response.usage.total_tokens

                set_span_attributes(span, span_updates)

                record_llm_metrics(
                    provider="openai",
                    model=response.model or self.model,
                    latency_seconds=latency_s,
                    status="success",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                return LLMResponse(
                    content=choice.message.content,
                    tool_calls=parsed_tool_calls,
                    model=response.model or self.model,
                    finish_reason=choice.finish_reason or "stop",
                )
            except OpenAIError as e:
                latency_s = time.perf_counter() - start_time
                record_span_error(span, e)
                set_span_attributes(span, {"status": "error"})
                record_llm_metrics(
                    provider="openai",
                    model=self.model,
                    latency_seconds=latency_s,
                    status="error",
                )
                logger.error("OpenAI API invocation failed: %s", type(e).__name__)
                raise RuntimeError(f"OpenAI service error: {type(e).__name__}") from e
            except Exception as e:
                latency_s = time.perf_counter() - start_time
                record_span_error(span, e)
                set_span_attributes(span, {"status": "error"})
                record_llm_metrics(
                    provider="openai",
                    model=self.model,
                    latency_seconds=latency_s,
                    status="error",
                )
                raise
