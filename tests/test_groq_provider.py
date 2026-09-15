"""Tests for Groq LLM Provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from src.app.llm.providers.groq import GroqLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall


def test_groq_initialization_requires_api_key() -> None:
    """Test that missing API key raises ValueError."""
    with pytest.raises(ValueError, match="Groq API key must be provided"):
        GroqLLMProvider(api_key="")


@pytest.mark.asyncio
async def test_groq_text_generation_normalization() -> None:
    """Test text response normalization into common LLMResponse schema."""
    provider = GroqLLMProvider(api_key="gsk-test-key", model="llama-3.3-70b-versatile")

    mock_choice = MagicMock()
    mock_choice.message.content = "Ultra-fast response from Groq!"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model = "llama-3.3-70b-versatile"

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_resp),
    ):
        response: LLMResponse = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert response.content == "Ultra-fast response from Groq!"
        assert response.tool_calls == []
        assert response.model == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_groq_tool_call_normalization() -> None:
    """Test tool-call normalization into common ToolCall schema."""
    provider = GroqLLMProvider(api_key="gsk-test-key", model="llama-3.3-70b-versatile")

    mock_tc = MagicMock()
    mock_tc.id = "call_groq_99"
    mock_tc.function.name = "calculator"
    mock_tc.function.arguments = '{"expression": "42 / 2"}'

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "tool_calls"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model = "llama-3.3-70b-versatile"

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_resp),
    ):
        response: LLMResponse = await provider.generate(
            messages=[{"role": "user", "content": "Calculate 42 / 2"}],
            tools=[{"name": "calculator", "description": "Calc", "parameters": {}}],
        )

        assert response.content is None
        assert len(response.tool_calls) == 1
        tc: ToolCall = response.tool_calls[0]
        assert tc.id == "call_groq_99"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "42 / 2"}


@pytest.mark.asyncio
async def test_groq_api_error_handling() -> None:
    """Test that API errors are wrapped cleanly."""
    provider = GroqLLMProvider(api_key="gsk-secret-key-12345", model="llama-3.3-70b-versatile")

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=OpenAIError("Groq service overloaded")),
    ):
        with pytest.raises(RuntimeError, match="Groq service error"):
            await provider.generate(messages=[{"role": "user", "content": "Hi"}])
