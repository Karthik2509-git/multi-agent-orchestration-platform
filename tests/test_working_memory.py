"""Unit tests for working memory and LangGraph checkpointer."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.memory.working_memory import get_checkpointer
from src.app.models.schemas.llm import LLMResponse
from src.app.orchestration.graph import build_orchestration_graph
from src.app.orchestration.state import OrchestrationState


@pytest.mark.asyncio
async def test_get_checkpointer_ephemeral_fallback():
    """Verify get_checkpointer returns MemorySaver in ephemeral / test mode."""
    settings = Settings(checkpoint_backend="memory")
    checkpointer = await get_checkpointer(settings, ephemeral=True)
    assert isinstance(checkpointer, MemorySaver)


@pytest.mark.asyncio
async def test_thread_state_checkpoint_and_recovery():
    """Verify state saved in thread_1 is checkpointed and accessible across invocations."""
    checkpointer = MemorySaver()
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            LLMResponse(content="Calculated value is 42.", model="mock-llm"),
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final synthesized output with 42.", model="mock-llm"),
        ]
    )

    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
    )

    thread_config = {"configurable": {"thread_id": "thread_alpha"}}

    initial_state: OrchestrationState = {
        "task": "Compute revenue metric",
        "messages": [{"role": "user", "content": "Compute revenue metric"}],
        "next_agent": "",
        "agent_results": {},
        "agents_used": [],
        "step_count": 0,
        "final_answer": "",
        "status": "pending",
        "metadata": {"user": "alice"},
        "memories_used": [],
    }

    # Step 1: Run graph on thread_alpha
    final_state = await graph.ainvoke(initial_state, config=thread_config)
    assert final_state["status"] == "completed"
    assert "42" in final_state["final_answer"]

    # Step 2: Checkpoint verification
    state_tuple = await checkpointer.aget_tuple(thread_config)
    assert state_tuple is not None
    saved_values = state_tuple.checkpoint["channel_values"]
    assert saved_values["status"] == "completed"
    assert "data" in saved_values["agents_used"]
    assert saved_values["metadata"]["user"] == "alice"


@pytest.mark.asyncio
async def test_independent_thread_isolation():
    """Verify thread A state does not bleed into thread B."""
    checkpointer = MemorySaver()
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Thread A output", model="mock-llm"),
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Thread B output", model="mock-llm"),
        ]
    )

    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
    )

    config_a = {"configurable": {"thread_id": "thread_a"}}
    config_b = {"configurable": {"thread_id": "thread_b"}}

    state_a: OrchestrationState = {
        "task": "Task for A",
        "messages": [{"role": "user", "content": "Task for A"}],
        "metadata": {"tenant": "alpha"},
    }
    state_b: OrchestrationState = {
        "task": "Task for B",
        "messages": [{"role": "user", "content": "Task for B"}],
        "metadata": {"tenant": "beta"},
    }

    res_a = await graph.ainvoke(state_a, config=config_a)
    res_b = await graph.ainvoke(state_b, config=config_b)

    assert "Thread A output" in res_a["final_answer"]
    assert "Thread B output" in res_b["final_answer"]

    tuple_a = await checkpointer.aget_tuple(config_a)
    tuple_b = await checkpointer.aget_tuple(config_b)

    assert tuple_a.checkpoint["channel_values"]["metadata"]["tenant"] == "alpha"
    assert tuple_b.checkpoint["channel_values"]["metadata"]["tenant"] == "beta"


@pytest.mark.asyncio
async def test_checkpointer_lifespan_lifecycle():
    """Verify init_checkpointer, get_checkpointer, and close_checkpointer manage lifecycle."""
    from src.app.memory.working_memory import (
        close_checkpointer,
        get_checkpointer,
        init_checkpointer,
        reset_checkpointer,
    )

    reset_checkpointer()
    settings = Settings(checkpoint_backend="memory")

    # 1. Startup initialization
    cp = await init_checkpointer(settings)
    assert isinstance(cp, MemorySaver)

    # 2. Get checkpointer reuses instance
    cp_reused = await get_checkpointer(settings)
    assert cp_reused is cp

    # 3. Shutdown
    await close_checkpointer()
    reset_checkpointer()


@pytest.mark.asyncio
async def test_postgres_checkpointer_failure_raises_without_fallback():
    """Verify that when checkpoint_backend=postgres, initialization failure raises RuntimeError."""
    from src.app.memory.working_memory import init_checkpointer, reset_checkpointer

    reset_checkpointer()
    # Configure an unreachable database endpoint
    settings = Settings(
        checkpoint_backend="postgres",
        database_url="postgresql://nonexistent_user:wrong_pwd@127.0.0.1:59999/nonexistent_db",
    )

    with pytest.raises(RuntimeError) as exc_info:
        await init_checkpointer(settings)

    assert "Failed to initialize PostgreSQL checkpointer" in str(exc_info.value)
    reset_checkpointer()
