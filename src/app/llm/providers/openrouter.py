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

        formatted_tools = None
        if tools:
            formatted_tools = [{"type": "function", "function": tool} for tool in tools]

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if formatted_tools:
                kwargs["tools"] = formatted_tools

            response = await self.client.chat.completions.create(**kwargs)
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

            return LLMResponse(
                content=choice.message.content,
                tool_calls=parsed_tool_calls,
                model=response.model or self.model,
                finish_reason=choice.finish_reason or "stop",
            )
        except OpenAIError as e:
            logger.error("OpenRouter API invocation failed: %s", type(e).__name__)
            raise RuntimeError(f"OpenRouter service error: {type(e).__name__}") from e
