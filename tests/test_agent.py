"""Tests for ToolCallingAgent using MockLLMProvider."""

from unittest.mock import patch

import pytest

from src.app.agents.tool_calling_agent import ToolCallingAgent
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool
from src.app.tools.registry import ToolRegistry


@pytest.fixture
def test_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        SafeHTTPGetTool(
            allowed_domains=["httpbin.org"],
            timeout=2.0,
            max_size_bytes=1000,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_agent_no_tool_call(test_registry: ToolRegistry) -> None:
    """Test agent answering a simple question directly without calling any tool."""
    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Paris is the capital of France.",
                tool_calls=[],
                model="mock-gpt-4o",
            )
        ]
    )

    agent = ToolCallingAgent(provider=provider, registry=test_registry)
    response = await agent.run("What is the capital of France?")

    assert response.status == "completed"
    assert response.answer == "Paris is the capital of France."
    assert len(response.tool_calls) == 0
    assert response.execution_time_seconds >= 0


@pytest.mark.asyncio
async def test_agent_calculator_tool_call(test_registry: ToolRegistry) -> None:
    """Test agent invoking calculator tool and synthesizing the result."""
    provider = MockLLMProvider(
        responses=[
            # Step 1: Model decides to call calculator
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="calculator",
                        arguments={"expression": "25 * 17"},
                    )
                ],
                model="mock-gpt-4o",
            ),
            # Step 2: Model consumes tool result and produces final answer
            LLMResponse(
                content="The product of 25 and 17 is 425.",
                tool_calls=[],
                model="mock-gpt-4o",
            ),
        ]
    )

    agent = ToolCallingAgent(provider=provider, registry=test_registry)
    response = await agent.run("Calculate 25 * 17")

    assert response.status == "completed"
    assert "425" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "calculator"
    assert response.tool_calls[0].success is True
    assert response.tool_calls[0].result["result"] == 425


@pytest.mark.asyncio
async def test_agent_http_tool_call(test_registry: ToolRegistry) -> None:
    """Test agent invoking safe HTTP GET tool."""
    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_http_1",
                        name="http_get",
                        arguments={"url": "https://httpbin.org/get"},
                    )
                ],
                model="mock-gpt-4o",
            ),
            LLMResponse(
                content="Successfully fetched data from httpbin.org",
                tool_calls=[],
                model="mock-gpt-4o",
            ),
        ]
    )

    agent = ToolCallingAgent(provider=provider, registry=test_registry)

    with patch("socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with patch("httpx.AsyncClient.stream") as mock_stream:
            from unittest.mock import AsyncMock

            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.url = "https://httpbin.org/get"
            mock_resp.headers = {}

            async def mock_aiter():
                yield b'{"origin": "1.2.3.4"}'

            mock_resp.aiter_bytes = mock_aiter
            mock_stream.return_value.__aenter__.return_value = mock_resp

            response = await agent.run("Fetch https://httpbin.org/get")

            assert response.status == "completed"
            assert "Successfully fetched" in response.answer
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].tool_name == "http_get"
            assert response.tool_calls[0].success is True


@pytest.mark.asyncio
async def test_agent_sequential_tool_calls(test_registry: ToolRegistry) -> None:
    """Test agent executing multiple sequential tool calls across iterations."""
    provider = MockLLMProvider(
        responses=[
            # Iteration 1: Call calculator 100 / 4
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="calculator",
                        arguments={"expression": "100 / 4"},
                    )
                ],
                model="mock-gpt-4o",
            ),
            # Iteration 2: Call calculator 25 + 50
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="calculator",
                        arguments={"expression": "25 + 50"},
                    )
                ],
                model="mock-gpt-4o",
            ),
            # Iteration 3: Final answer
            LLMResponse(
                content="The final result is 75.",
                tool_calls=[],
                model="mock-gpt-4o",
            ),
        ]
    )

    agent = ToolCallingAgent(provider=provider, registry=test_registry, max_iterations=5)
    response = await agent.run("Do two calculations")

    assert response.status == "completed"
    assert response.answer == "The final result is 75."
    assert len(response.tool_calls) == 2


@pytest.mark.asyncio
async def test_agent_max_iterations_protection(test_registry: ToolRegistry) -> None:
    """Test agent halts gracefully when hitting max iterations limit."""
    # Endless tool call responses
    provider = MockLLMProvider()

    def endless_handler(messages, tools):
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="endless_call",
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                )
            ],
            model="mock-gpt-4o",
        )

    provider.handler = endless_handler

    agent = ToolCallingAgent(provider=provider, registry=test_registry, max_iterations=3)
    response = await agent.run("Run endless calculation")

    assert response.status == "max_iterations_reached"
    assert "maximum iteration limit" in response.answer
    assert len(response.tool_calls) == 3


@pytest.mark.asyncio
async def test_agent_tool_error_recovery(test_registry: ToolRegistry) -> None:
    """Test agent recovers when a tool returns an error."""
    provider = MockLLMProvider(
        responses=[
            # Step 1: Model requests division by zero
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="div_zero",
                        name="calculator",
                        arguments={"expression": "10 / 0"},
                    )
                ],
                model="mock-gpt-4o",
            ),
            # Step 2: Model acknowledges error and explains to user
            LLMResponse(
                content="Cannot divide 10 by zero as it is mathematically undefined.",
                tool_calls=[],
                model="mock-gpt-4o",
            ),
        ]
    )

    agent = ToolCallingAgent(provider=provider, registry=test_registry)
    response = await agent.run("Calculate 10 / 0")

    assert response.status == "completed"
    assert "Cannot divide" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].success is False
    assert "Division by zero" in response.tool_calls[0].error


@pytest.mark.asyncio
async def test_agent_llm_failure_handling(test_registry: ToolRegistry) -> None:
    """Test agent gracefully handles LLM provider exceptions."""
    provider = MockLLMProvider()

    def failing_handler(messages, tools):
        raise RuntimeError("OpenAI connection timed out")

    provider.handler = failing_handler

    agent = ToolCallingAgent(provider=provider, registry=test_registry)
    response = await agent.run("What is 1 + 1?")

    assert response.status == "error"
    assert "OpenAI connection timed out" in response.answer
