"""Tests verifying ToolCallingAgent is completely provider-agnostic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.agents.tool_calling_agent import ToolCallingAgent
from src.app.llm.providers.gemini import GeminiLLMProvider
from src.app.llm.providers.groq import GroqLLMProvider
from src.app.llm.providers.openrouter import OpenRouterLLMProvider
from src.app.tools.calculator import CalculatorTool
from src.app.tools.registry import ToolRegistry


@pytest.fixture
def agent_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


@pytest.mark.asyncio
async def test_agent_with_openrouter_provider(agent_registry: ToolRegistry) -> None:
    """Verify ToolCallingAgent executes calculator tool seamlessly with OpenRouter."""
    provider = OpenRouterLLMProvider(api_key="sk-or-test", model="openrouter/free")

    # Step 1: OpenRouter produces calculator tool call
    mock_tc = MagicMock()
    mock_tc.id = "call_or_calc"
    mock_tc.function.name = "calculator"
    mock_tc.function.arguments = '{"expression": "50 + 50"}'

    choice_1 = MagicMock()
    choice_1.message.content = None
    choice_1.message.tool_calls = [mock_tc]
    choice_1.finish_reason = "tool_calls"
    resp_1 = MagicMock(choices=[choice_1], model="openrouter/free")

    # Step 2: OpenRouter consumes tool result and answers
    choice_2 = MagicMock()
    choice_2.message.content = "50 + 50 equals 100."
    choice_2.message.tool_calls = None
    choice_2.finish_reason = "stop"
    resp_2 = MagicMock(choices=[choice_2], model="openrouter/free")

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=[resp_1, resp_2]),
    ):
        agent = ToolCallingAgent(provider=provider, registry=agent_registry)
        response = await agent.run("Calculate 50 + 50")

        assert response.status == "completed"
        assert "100" in response.answer
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "calculator"
        assert response.tool_calls[0].result["result"] == 100


@pytest.mark.asyncio
async def test_agent_with_gemini_provider(agent_registry: ToolRegistry) -> None:
    """Verify ToolCallingAgent executes calculator tool seamlessly with Gemini."""
    provider = GeminiLLMProvider(api_key="gemini-key-test", model="gemini-2.5-flash")

    # Step 1: Gemini produces FunctionCall
    fc = MagicMock()
    fc.name = "calculator"
    fc.args = {"expression": "20 * 5"}

    gemini_resp_1 = MagicMock(text=None, function_calls=[fc])

    # Step 2: Gemini produces final answer text
    gemini_resp_2 = MagicMock(text="20 * 5 is 100.", function_calls=None)

    with patch.object(
        provider.client.aio.models,
        "generate_content",
        new=AsyncMock(side_effect=[gemini_resp_1, gemini_resp_2]),
    ):
        agent = ToolCallingAgent(provider=provider, registry=agent_registry)
        response = await agent.run("Calculate 20 * 5")

        assert response.status == "completed"
        assert "100" in response.answer
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "calculator"
        assert response.tool_calls[0].result["result"] == 100


@pytest.mark.asyncio
async def test_agent_with_groq_provider(agent_registry: ToolRegistry) -> None:
    """Verify ToolCallingAgent executes calculator tool seamlessly with Groq."""
    provider = GroqLLMProvider(api_key="gsk-test", model="llama-3.3-70b-versatile")

    mock_tc = MagicMock()
    mock_tc.id = "call_groq_1"
    mock_tc.function.name = "calculator"
    mock_tc.function.arguments = '{"expression": "1000 / 10"}'

    choice_1 = MagicMock()
    choice_1.message.content = None
    choice_1.message.tool_calls = [mock_tc]
    choice_1.finish_reason = "tool_calls"
    resp_1 = MagicMock(choices=[choice_1], model="llama-3.3-70b-versatile")

    choice_2 = MagicMock()
    choice_2.message.content = "1000 divided by 10 is 100."
    choice_2.message.tool_calls = None
    choice_2.finish_reason = "stop"
    resp_2 = MagicMock(choices=[choice_2], model="llama-3.3-70b-versatile")

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=[resp_1, resp_2]),
    ):
        agent = ToolCallingAgent(provider=provider, registry=agent_registry)
        response = await agent.run("Calculate 1000 / 10")

        assert response.status == "completed"
        assert "100" in response.answer
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "calculator"
        assert response.tool_calls[0].result["result"] == 100
