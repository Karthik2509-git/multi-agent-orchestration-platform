"""Tests for specialized worker agents (ResearchAgent, DataAgent, CodeAgent, FinalAgent)."""

import pytest

from src.app.agents.code_agent import CodeAgent
from src.app.agents.data_agent import DataAgent
from src.app.agents.final_agent import FinalAgent
from src.app.agents.research_agent import ResearchAgent
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.orchestration.state import OrchestrationState
from src.app.tools.base import ToolResult
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool


@pytest.mark.asyncio
async def test_research_agent_synthesizes_without_url():
    """Verify ResearchAgent synthesizes research topics directly with LLM."""
    mock_provider = MockLLMProvider(
        responses=[LLMResponse(content="Quantum computing uses qubits utilizing superposition.")]
    )
    agent = ResearchAgent(provider=mock_provider)
    state: OrchestrationState = {"task": "Explain quantum computing fundamentals"}

    result = await agent.run(task="Explain quantum computing fundamentals", state=state)

    assert result.agent == "research"
    assert result.status == "success"
    assert "qubits" in result.result


@pytest.mark.asyncio
async def test_research_agent_fetches_permitted_url(monkeypatch):
    """Verify ResearchAgent fetches permitted URLs if detected in task."""
    mock_http_tool = SafeHTTPGetTool(allowed_domains=["example.com"])

    async def mock_execute(url: str):
        return ToolResult(
            success=True,
            data={"url": url, "content": "Special report on AI advances 2024."},
        )

    monkeypatch.setattr(mock_http_tool, "execute", mock_execute)

    mock_provider = MockLLMProvider(
        responses=[LLMResponse(content="Based on the report, AI advances occurred in 2024.")]
    )
    agent = ResearchAgent(provider=mock_provider, http_tool=mock_http_tool)
    state: OrchestrationState = {"task": "Review https://example.com/report for AI updates"}

    result = await agent.run(task="Review https://example.com/report for AI updates", state=state)

    assert result.agent == "research"
    assert result.status == "success"
    assert result.metadata.get("url_inspected") == "https://example.com/report"
    # Ensure URL content was incorporated in the LLM prompt
    last_prompt = mock_provider.history[0]["messages"][-1]["content"]
    assert "Special report on AI advances" in last_prompt


@pytest.mark.asyncio
async def test_data_agent_with_calculator():
    """Verify DataAgent leverages CalculatorTool to evaluate expressions."""
    mock_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="The calculation result is 42.",
                tool_calls=[
                    ToolCall(
                        id="call_calc",
                        name="calculator",
                        arguments={"expression": "6 * 7"},
                    )
                ],
            )
        ]
    )
    calculator = CalculatorTool()
    agent = DataAgent(provider=mock_provider, calculator=calculator)
    state: OrchestrationState = {"task": "Calculate 6 * 7"}

    result = await agent.run(task="Calculate 6 * 7", state=state)

    assert result.agent == "data"
    assert result.status == "success"
    assert "Evaluated 6 * 7: 42" in result.result
    assert result.metadata.get("calculations_performed") == 1


@pytest.mark.asyncio
async def test_code_agent_generates_code_safely():
    """Verify CodeAgent generates code without executing or running shell."""
    code_content = "def add(a: int, b: int) -> int:\n    return a + b"
    mock_provider = MockLLMProvider(responses=[LLMResponse(content=code_content)])
    agent = CodeAgent(provider=mock_provider)
    state: OrchestrationState = {"task": "Write an add function in Python"}

    result = await agent.run(task="Write an add function in Python", state=state)

    assert result.agent == "code"
    assert result.status == "success"
    assert "def add(a: int, b: int)" in result.result
    assert result.metadata.get("code_execution_attempted") is False


@pytest.mark.asyncio
async def test_final_agent_synthesizes_multiple_results():
    """Verify FinalAgent aggregates multi-agent findings into a final answer."""
    mock_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Final report: Apple grew 15% in revenue with strong mobile performance."
            )
        ]
    )
    agent = FinalAgent(provider=mock_provider)
    state: OrchestrationState = {
        "task": "Comprehensive Apple financial overview",
        "agent_results": {
            "research": "Apple annual report revealed 15% revenue expansion.",
            "data": "CAGR was calculated at 7.2%.",
        },
    }

    result = await agent.run(task="Comprehensive Apple financial overview", state=state)

    assert result.agent == "final"
    assert result.status == "success"
    assert "Final report" in result.result
    assert "research" in result.metadata.get("aggregated_agents", [])
    assert "data" in result.metadata.get("aggregated_agents", [])
