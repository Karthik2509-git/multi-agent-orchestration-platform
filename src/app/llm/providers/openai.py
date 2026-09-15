"""OpenAI LLM Provider implementation using official AsyncOpenAI SDK."""

import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAIError

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall

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

            return LLMResponse(
                content=choice.message.content,
                tool_calls=parsed_tool_calls,
                model=response.model,
                finish_reason=choice.finish_reason or "stop",
            )
        except OpenAIError as e:
            logger.error("OpenAI API invocation failed: %s", type(e).__name__)
            raise RuntimeError(f"OpenAI service error: {type(e).__name__}") from e
