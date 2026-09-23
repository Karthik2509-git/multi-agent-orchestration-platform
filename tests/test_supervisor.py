"""Tests for SupervisorAgent and strict route parsing."""

import pytest

from src.app.agents.supervisor import SupervisorAgent
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse
from src.app.orchestration.state import OrchestrationState


@pytest.mark.asyncio
async def test_supervisor_valid_routing():
    """Verify standard valid JSON routing for each allowed worker."""
    for agent in ["research", "data", "code", "final"]:
        mock_provider = MockLLMProvider(
            responses=[
                LLMResponse(content=f'{{"next_agent": "{agent}", "reasoning": "Need {agent}"}}')
            ]
        )
        supervisor = SupervisorAgent(provider=mock_provider)
        state: OrchestrationState = {"task": f"Please do {agent} work", "step_count": 0}
        decision = await supervisor.decide_route(state)

        assert decision.next_agent == agent
        assert decision.reasoning == f"Need {agent}"


@pytest.mark.asyncio
async def test_supervisor_markdown_fenced_json():
    """Verify that json within markdown code blocks is properly stripped and parsed."""
    fenced_output = """```json
{
  "next_agent": "data",
  "reasoning": "Compute metrics"
}
```"""
    mock_provider = MockLLMProvider(responses=[LLMResponse(content=fenced_output)])
    supervisor = SupervisorAgent(provider=mock_provider)
    state: OrchestrationState = {"task": "Calculate CAGR", "step_count": 0}
    decision = await supervisor.decide_route(state)

    assert decision.next_agent == "data"
    assert decision.reasoning == "Compute metrics"


@pytest.mark.asyncio
async def test_supervisor_rejects_unauthorized_agent():
    """Verify unauthorized routes (like 'browser' or 'system') fall back to 'final'."""
    unauthorized_json = '{"next_agent": "browser", "reasoning": "Browse the web"}'
    mock_provider = MockLLMProvider(responses=[LLMResponse(content=unauthorized_json)])
    supervisor = SupervisorAgent(provider=mock_provider)
    state: OrchestrationState = {"task": "Surf the internet", "step_count": 0}
    decision = await supervisor.decide_route(state)

    assert decision.next_agent == "final"
    assert "defaulted to final" in (decision.reasoning or "").lower()


@pytest.mark.asyncio
async def test_supervisor_malformed_json_fallback():
    """Verify that malformed or garbage output safely defaults to 'final'."""
    mock_provider = MockLLMProvider(
        responses=[LLMResponse(content="I think we should just give up here!")]
    )
    supervisor = SupervisorAgent(provider=mock_provider)
    state: OrchestrationState = {"task": "Something confusing", "step_count": 0}
    decision = await supervisor.decide_route(state)

    assert decision.next_agent == "final"
    assert "defaulted to final" in (decision.reasoning or "").lower()


@pytest.mark.asyncio
async def test_supervisor_empty_response_fallback():
    """Verify that empty or None model responses fall back to 'final'."""
    mock_provider = MockLLMProvider(responses=[LLMResponse(content=None)])
    supervisor = SupervisorAgent(provider=mock_provider)
    state: OrchestrationState = {"task": "Empty test", "step_count": 0}
    decision = await supervisor.decide_route(state)

    assert decision.next_agent == "final"


@pytest.mark.asyncio
async def test_supervisor_includes_agent_history_in_prompt():
    """Verify that completed worker results are fed into the supervisor context."""
    mock_provider = MockLLMProvider(
        responses=[LLMResponse(content='{"next_agent": "final", "reasoning": "Done"}')]
    )
    supervisor = SupervisorAgent(provider=mock_provider)
    state: OrchestrationState = {
        "task": "Perform research then calculate",
        "agent_results": {
            "research": "Found company growth was 15 percent in 2022 and 20 percent in 2023."
        },
        "step_count": 1,
    }
    decision = await supervisor.decide_route(state)
    assert decision.next_agent == "final"

    # Check generated prompt messages in mock provider
    last_call = mock_provider.history[0]
    user_prompt = last_call["messages"][-1]["content"]
    assert "Found company growth was 15 percent" in user_prompt
    assert "Current Step: 1" in user_prompt
