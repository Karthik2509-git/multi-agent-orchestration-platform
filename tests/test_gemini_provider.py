"""Tests for Google Gemini LLM Provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors

from src.app.llm.providers.gemini import GeminiLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall


def test_gemini_initialization_requires_api_key() -> None:
    """Test that missing API key raises ValueError."""
    with pytest.raises(ValueError, match="Gemini API key must be provided"):
        GeminiLLMProvider(api_key="")


def test_gemini_message_conversion() -> None:
    """Test converting internal messages to Gemini Content and system instruction."""
    provider = GeminiLLMProvider(api_key="gemini-test-key")

    internal_messages = [
        {"role": "system", "content": "You are a helpful math agent."},
        {"role": "user", "content": "Calculate 10 + 20"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "name": "calculator",
                    "arguments": {"expression": "10 + 20"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "calculator",
            "content": '{"result": 30}',
        },
    ]

    system_instruction, contents = provider._convert_messages(internal_messages)

    assert system_instruction == "You are a helpful math agent."
    assert len(contents) == 3

    # User message
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Calculate 10 + 20"

    # Assistant message with function call
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "calculator"
    assert contents[1].parts[0].function_call.args == {"expression": "10 + 20"}

    # Tool result message
    assert contents[2].role == "user"
    assert contents[2].parts[0].function_response.name == "calculator"
    assert contents[2].parts[0].function_response.response == {"response": {"result": 30}}


def test_gemini_tool_conversion() -> None:
    """Test converting OpenAI JSON Schema tools to Gemini Tool declarations."""
    provider = GeminiLLMProvider(api_key="gemini-test-key")

    tools = [
        {
            "name": "calculator",
            "description": "Safe calculator",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]

    gemini_tools = provider._convert_tools(tools)
    assert gemini_tools is not None
    assert len(gemini_tools) == 1
    assert len(gemini_tools[0].function_declarations) == 1
    decl = gemini_tools[0].function_declarations[0]
    assert decl.name == "calculator"
    assert decl.description == "Safe calculator"


@pytest.mark.asyncio
async def test_gemini_text_generation_normalization() -> None:
    """Test normalizing Gemini text response into common LLMResponse."""
    provider = GeminiLLMProvider(api_key="gemini-test-key", model="gemini-2.5-flash")

    mock_resp = MagicMock()
    mock_resp.text = "Gemini answer: 30"
    mock_resp.function_calls = None

    with patch.object(
        provider.client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_resp),
    ):
        response: LLMResponse = await provider.generate(
            messages=[{"role": "user", "content": "What is 10 + 20?"}]
        )

        assert response.content == "Gemini answer: 30"
        assert response.tool_calls == []
        assert response.model == "gemini-2.5-flash"
        assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_gemini_tool_call_normalization() -> None:
    """Test normalizing Gemini function calls into common ToolCall objects."""
    provider = GeminiLLMProvider(api_key="gemini-test-key", model="gemini-2.5-flash")

    mock_fc = MagicMock()
    mock_fc.name = "calculator"
    mock_fc.args = {"expression": "25 * 4"}

    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.function_calls = [mock_fc]

    with patch.object(
        provider.client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_resp),
    ):
        response: LLMResponse = await provider.generate(
            messages=[{"role": "user", "content": "Calculate 25 * 4"}],
            tools=[{"name": "calculator", "description": "Calc", "parameters": {}}],
        )

        assert len(response.tool_calls) == 1
        tc: ToolCall = response.tool_calls[0]
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "25 * 4"}
        assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_gemini_api_error_handling() -> None:
    """Test that Gemini API errors are cleanly caught and wrapped."""
    provider = GeminiLLMProvider(api_key="gemini-test-key")

    with patch.object(
        provider.client.aio.models,
        "generate_content",
        new=AsyncMock(side_effect=errors.APIError(500, {"error": "Gemini quota exceeded"})),
    ):
        with pytest.raises(RuntimeError, match="Gemini service error"):
            await provider.generate(messages=[{"role": "user", "content": "Hi"}])
