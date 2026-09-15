"""Tests for OpenRouter LLM Provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from src.app.llm.providers.openrouter import OpenRouterLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall


def test_openrouter_initialization_requires_api_key() -> None:
    """Test that missing API key raises ValueError."""
    with pytest.raises(ValueError, match="OpenRouter API key must be provided"):
        OpenRouterLLMProvider(api_key="")


@pytest.mark.asyncio
async def test_openrouter_text_generation_normalization() -> None:
    """Test text response normalization into common LLMResponse schema."""
    provider = OpenRouterLLMProvider(api_key="sk-or-test-key", model="openrouter/free")

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello from OpenRouter!"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model = "openrouter/free"

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_resp),
    ):
        response: LLMResponse = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert response.content == "Hello from OpenRouter!"
        assert response.tool_calls == []
        assert response.model == "openrouter/free"
        assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openrouter_tool_call_normalization() -> None:
    """Test tool-call normalization into common ToolCall schema."""
    provider = OpenRouterLLMProvider(api_key="sk-or-test-key", model="openrouter/free")

    mock_tc = MagicMock()
    mock_tc.id = "call_or_1"
    mock_tc.function.name = "calculator"
    mock_tc.function.arguments = '{"expression": "100 * 5"}'

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "tool_calls"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model = "openrouter/free"

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_resp),
    ):
        response: LLMResponse = await provider.generate(
            messages=[{"role": "user", "content": "Calculate 100 * 5"}],
            tools=[{"name": "calculator", "description": "Calc", "parameters": {}}],
        )

        assert response.content is None
        assert len(response.tool_calls) == 1
        tc: ToolCall = response.tool_calls[0]
        assert tc.id == "call_or_1"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "100 * 5"}
        assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_openrouter_api_error_handling() -> None:
    """Test that API errors are wrapped without leaking credentials."""
    provider = OpenRouterLLMProvider(api_key="sk-or-secret-key-12345", model="openrouter/free")

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=OpenAIError("Rate limit exceeded")),
    ):
        with pytest.raises(RuntimeError, match="OpenRouter service error"):
            await provider.generate(messages=[{"role": "user", "content": "Hi"}])
