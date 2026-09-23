"""Tests for OrchestrationState definition and typing."""

from src.app.orchestration.state import OrchestrationState


def test_orchestration_state_structure():
    """Verify that OrchestrationState supports all expected state fields."""
    state: OrchestrationState = {
        "task": "Test research task",
        "messages": [{"role": "user", "content": "Test research task"}],
        "next_agent": "research",
        "agent_results": {"research": "Found results"},
        "agents_used": ["research"],
        "step_count": 1,
        "final_answer": "Final synthesis",
        "status": "completed",
        "metadata": {"custom_key": "custom_value"},
    }

    assert state["task"] == "Test research task"
    assert state["next_agent"] == "research"
    assert state["step_count"] == 1
    assert "research" in state["agents_used"]
    assert state["agent_results"]["research"] == "Found results"
    assert state["status"] == "completed"


def test_orchestration_state_empty_initialization():
    """Verify that OrchestrationState can be partially initialized without error."""
    state: OrchestrationState = {
        "task": "Minimal task",
    }
    assert state.get("step_count", 0) == 0
    assert state.get("agent_results", {}) == {}
