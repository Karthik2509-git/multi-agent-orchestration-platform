"""Tests for the LangGraph multi-agent orchestration graph."""

import pytest

from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse
from src.app.orchestration.graph import build_orchestration_graph
from src.app.orchestration.state import OrchestrationState


@pytest.mark.asyncio
async def test_graph_direct_to_final():
    """Verify flow where supervisor immediately chooses 'final'."""
    mock_provider = MockLLMProvider(
        responses=[
            # Supervisor decision
            LLMResponse(content='{"next_agent": "final", "reasoning": "Simple greeting"}'),
            # Final agent response
            LLMResponse(content="Hello! How can I assist you today?"),
        ]
    )

    graph = build_orchestration_graph(provider=mock_provider)
    initial_state: OrchestrationState = {
        "task": "Say hello",
        "step_count": 0,
        "agents_used": [],
        "agent_results": {},
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["status"] == "completed"
    assert "Hello! How can I assist you" in final_state["final_answer"]
    assert "final" in final_state["agents_used"]
    assert "research" not in final_state["agents_used"]


@pytest.mark.asyncio
async def test_graph_single_specialist_data_flow():
    """Verify flow: Supervisor -> Data -> Supervisor -> Final -> END."""
    mock_provider = MockLLMProvider(
        responses=[
            # Supervisor decides data
            LLMResponse(content='{"next_agent": "data", "reasoning": "Needs calculation"}'),
            # Data agent executes
            LLMResponse(content="Calculated sum is 100."),
            # Supervisor decides final
            LLMResponse(content='{"next_agent": "final", "reasoning": "Calculation done"}'),
            # Final agent responds
            LLMResponse(content="The final result of the sum is 100."),
        ]
    )

    graph = build_orchestration_graph(provider=mock_provider)
    initial_state: OrchestrationState = {
        "task": "Add numbers together",
        "step_count": 0,
        "agents_used": [],
        "agent_results": {},
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["status"] == "completed"
    assert "The final result" in final_state["final_answer"]
    assert "data" in final_state["agents_used"]
    assert "final" in final_state["agents_used"]
    assert "data" in final_state["agent_results"]


@pytest.mark.asyncio
async def test_graph_multi_specialist_sequential_flow():
    """Verify flow: Supervisor -> Research -> Supervisor -> Code -> Supervisor -> Final -> END."""
    mock_provider = MockLLMProvider(
        responses=[
            # 1. Supervisor routes to research
            LLMResponse(
                content=(
                    '{"next_agent": "research", "reasoning": "Need research on sorting algorithms"}'
                )
            ),
            # 2. Research agent executes
            LLMResponse(
                content="Quicksort has average complexity O(n log n) and worst case O(n^2)."
            ),
            # 3. Supervisor routes to code
            LLMResponse(
                content='{"next_agent": "code", "reasoning": "Need implementation of quicksort"}'
            ),
            # 4. Code agent executes
            LLMResponse(content="def quicksort(arr): return arr if len(arr) <= 1 else ..."),
            # 5. Supervisor routes to final
            LLMResponse(
                content=(
                    '{"next_agent": "final", "reasoning": "Both research and code are complete"}'
                )
            ),
            # 6. Final agent synthesizes
            LLMResponse(
                content=(
                    "Quicksort operates in O(n log n) time. Here is the implementation:\n"
                    "def quicksort..."
                )
            ),
        ]
    )

    graph = build_orchestration_graph(provider=mock_provider)
    initial_state: OrchestrationState = {
        "task": "Explain quicksort complexity and implement it in Python",
        "step_count": 0,
        "agents_used": [],
        "agent_results": {},
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["status"] == "completed"
    assert "research" in final_state["agents_used"]
    assert "code" in final_state["agents_used"]
    assert "final" in final_state["agents_used"]
    assert "research" in final_state["agent_results"]
    assert "code" in final_state["agent_results"]
    assert "Quicksort operates in O(n log n)" in final_state["final_answer"]


@pytest.mark.asyncio
async def test_graph_step_ceiling_terminates_loop():
    """Verify infinite loop between supervisor and workers terminates at step ceiling."""

    def dynamic_handler(messages, tools=None):
        system_content = messages[0]["content"] if messages else ""
        if "AI Orchestration Supervisor" in system_content:
            # Supervisor perpetually loops to 'data'
            return LLMResponse(content='{"next_agent": "data", "reasoning": "Keep calculating"}')
        elif "Data Analysis Agent" in system_content:
            return LLMResponse(content="Intermediate calculation step.")
        else:
            # Final agent
            return LLMResponse(content="Forced termination final answer.")

    mock_provider = MockLLMProvider(handler=dynamic_handler)
    graph = build_orchestration_graph(provider=mock_provider)

    initial_state: OrchestrationState = {
        "task": "Endless loop test",
        "step_count": 0,
        "agents_used": [],
        "agent_results": {},
    }

    final_state = await graph.ainvoke(initial_state)

    # Workflow must terminate
    assert final_state["status"] == "completed"
    assert "Forced termination" in final_state["final_answer"]
    assert final_state.get("metadata", {}).get("terminated_due_to_limit") is True
