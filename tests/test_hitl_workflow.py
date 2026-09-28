"""Unit and integration tests for LangGraph native HITL interrupt and Command resume."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.app.hitl.models import ApprovalDecision, HITLResponse
from src.app.hitl.policies import EscalationPolicy
from src.app.hitl.service import HITLService
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse
from src.app.orchestration.graph import build_orchestration_graph
from src.app.orchestration.state import OrchestrationState


@pytest.fixture
def hitl_setup():
    """Fixture providing compiled graph with checkpointer and HITLService."""
    checkpointer = MemorySaver()
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final automated answer after approval.", model="mock-llm"),
        ]
    )
    policy = EscalationPolicy()
    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
        hitl_policy=policy,
    )
    service = HITLService(checkpointer=checkpointer, policy=policy)
    return graph, checkpointer, service


@pytest.mark.asyncio
async def test_hitl_interrupt_and_approve_resume(hitl_setup):
    """Verify task pauses on interrupt, exposes pending approval, and completes upon APPROVE."""
    graph, checkpointer, service = hitl_setup
    thread_id = "thread_approve_test"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: OrchestrationState = {
        "task": "Execute sensitive financial transfer",
        "messages": [{"role": "user", "content": "Execute sensitive financial transfer"}],
        "metadata": {"requires_human_approval": True},
        "step_count": 0,
        "agents_used": [],
    }

    # 1. Run until approval gate triggers interrupt
    await graph.ainvoke(initial_state, config=config)

    # 2. Inspect pending approval
    pending = await service.get_pending_approval(thread_id)
    assert pending is not None
    assert pending.thread_id == thread_id
    assert "explicitly flagged" in pending.reason

    # 3. Submit APPROVE decision
    approval_resp = HITLResponse(decision=ApprovalDecision.APPROVE, feedback="Authorized by CFO.")
    final_state = await service.resume_thread(thread_id, approval_resp, graph)

    assert final_state["status"] == "completed"
    assert "Final automated answer" in final_state["final_answer"]


@pytest.mark.asyncio
async def test_hitl_reject_resume(hitl_setup):
    """Verify human REJECT decision terminates workflow cleanly with feedback."""
    graph, checkpointer, service = hitl_setup
    thread_id = "thread_reject_test"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: OrchestrationState = {
        "task": "Execute sensitive database wipe",
        "messages": [{"role": "user", "content": "Execute sensitive database wipe"}],
        "metadata": {"requires_human_approval": True},
        "step_count": 0,
        "agents_used": [],
    }

    await graph.ainvoke(initial_state, config=config)

    # Submit REJECT decision
    reject_resp = HITLResponse(
        decision=ApprovalDecision.REJECT, feedback="Operation denied: safety violation."
    )
    final_state = await service.resume_thread(thread_id, reject_resp, graph)

    assert final_state["status"] == "error"
    assert "aborted by human reviewer" in final_state["final_answer"]
    assert "safety violation" in final_state["final_answer"]


@pytest.mark.asyncio
async def test_hitl_takeover_resume(hitl_setup):
    """Verify human TAKE_OVER decision directly injects human answer and completes."""
    graph, checkpointer, service = hitl_setup
    thread_id = "thread_takeover_test"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: OrchestrationState = {
        "task": "Complex edge case analysis",
        "messages": [{"role": "user", "content": "Complex edge case analysis"}],
        "metadata": {"requires_human_approval": True},
        "step_count": 0,
        "agents_used": [],
    }

    await graph.ainvoke(initial_state, config=config)

    # Submit TAKE_OVER decision
    takeover_resp = HITLResponse(
        decision=ApprovalDecision.TAKE_OVER,
        override_output="Expert human verified answer: System is optimal.",
    )
    final_state = await service.resume_thread(thread_id, takeover_resp, graph)

    assert final_state["status"] == "completed"
    assert final_state["final_answer"] == "Expert human verified answer: System is optimal."
    assert final_state["metadata"]["human_decision"] == "take_over"


@pytest.mark.asyncio
async def test_hitl_modify_resume_end_to_end():
    """Verify human MODIFY decision updates action parameters and resumes to completion."""
    checkpointer = MemorySaver()
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(
                content="Completed execution with revised tax rate of 25%.", model="mock-llm"
            ),
        ]
    )
    policy = EscalationPolicy()
    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
        hitl_policy=policy,
    )
    service = HITLService(checkpointer=checkpointer, policy=policy)

    thread_id = "thread_modify_test"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: OrchestrationState = {
        "task": "Process tax report for 2025",
        "messages": [{"role": "user", "content": "Process tax report for 2025"}],
        "metadata": {"requires_human_approval": True},
        "step_count": 0,
        "agents_used": [],
    }

    # 1. Run until interrupt
    await graph.ainvoke(initial_state, config=config)

    # 2. Verify pending interrupt
    pending = await service.get_pending_approval(thread_id)
    assert pending is not None
    assert pending.thread_id == thread_id

    # 3. Resume with MODIFY payload
    modify_resp = HITLResponse(
        decision=ApprovalDecision.MODIFY,
        feedback="Adjust calculation rate to 25%",
        modified_action={"task": "Process tax report for 2025 at 25% rate", "tax_rate": 0.25},
    )
    final_state = await service.resume_thread(thread_id, modify_resp, graph)

    # 4. Verify graph resumed, applied parameters, and finished
    assert final_state["status"] == "completed"
    assert final_state["metadata"]["human_decision"] == "modified"
    assert final_state["metadata"]["modified_action"]["tax_rate"] == 0.25
    assert final_state["metadata"]["human_feedback"] == "Adjust calculation rate to 25%"
    assert final_state["task"] == "Process tax report for 2025 at 25% rate"
    assert "25%" in final_state["final_answer"]
