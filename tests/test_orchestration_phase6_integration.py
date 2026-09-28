"""End-to-end integration tests for Phase 6: Orchestration + Memory + HITL + RAG + MCP."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings
from src.app.hitl.models import ApprovalDecision, HITLResponse
from src.app.hitl.service import HITLService
from src.app.llm.providers.mock import MockLLMProvider
from src.app.memory.models import MemoryType
from src.app.memory.service import MemoryService
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore
from src.app.models.schemas.llm import LLMResponse
from src.app.orchestration.graph import build_orchestration_graph
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.services.orchestration_service import run_orchestrated_task


@pytest.mark.asyncio
async def test_orchestration_with_memory_retrieval_and_hitl():
    """Verify prior memory is retrieved into supervisor context, and approval gate functions."""
    settings = Settings(
        memory_enabled=True,
        hitl_enabled=True,
        checkpoint_backend="memory",
    )
    checkpointer = MemorySaver()
    embedder = MockEmbeddingProvider(dimension=16)

    # 1. Setup Long-Term Memory Store with pre-existing memory
    mem_store = ChromaMemoryStore(
        collection_name="test_integ_mem_kb",
        embedding_provider=embedder,
        ephemeral=True,
    )
    memory_service = MemoryService(store=mem_store, settings=settings)

    await memory_service.add_memory(
        content="User constraint: always verify financial metrics with calculator tool.",
        scope_id="user_123",
        memory_type=MemoryType.USER_PREFERENCE,
        importance=0.9,
    )

    # 2. Setup Mock LLM
    mock_llm = MockLLMProvider(
        responses=[
            # Supervisor routes to data agent
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            # Data agent calculates
            LLMResponse(content="Data verified: Result is 2500.", model="mock-llm"),
            # Supervisor routes to final
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            # Final agent synthesizes answer
            LLMResponse(
                content="Final synthesized financial report: Result is 2500.", model="mock-llm"
            ),
        ]
    )

    # 3. Run orchestrated task with require_human_review = True
    thread_id = "phase6_integ_thread_1"
    response = await run_orchestrated_task(
        task="Calculate quarterly metric and produce report",
        settings=settings,
        thread_id=thread_id,
        scope_id="user_123",
        require_human_review=True,
        provider=mock_llm,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )

    # Should pause on approval gate
    assert response.status == "interrupted"
    assert response.thread_id == thread_id
    assert len(response.memories_used) >= 1
    assert "User constraint" in response.memories_used[0]["content"]

    # 4. Resume task with APPROVE
    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )
    hitl_service = HITLService(checkpointer=checkpointer, settings=settings)
    final_res = await hitl_service.resume_thread(
        thread_id=thread_id,
        response=HITLResponse(decision=ApprovalDecision.APPROVE),
        graph=graph,
    )

    assert final_res["status"] == "completed"
    assert "Result is 2500" in final_res["final_answer"]


@pytest.mark.asyncio
async def test_orchestration_hitl_modify_and_memory_extraction():
    """Verify MODIFY workflow applies parameters and extracts memories only upon completion."""
    settings = Settings(
        memory_enabled=True,
        hitl_enabled=True,
        checkpoint_backend="memory",
    )
    checkpointer = MemorySaver()
    embedder = MockEmbeddingProvider(dimension=16)

    mem_store = ChromaMemoryStore(
        collection_name="test_integ_modify_mem",
        embedding_provider=embedder,
        ephemeral=True,
    )
    memory_service = MemoryService(store=mem_store, settings=settings)

    mock_llm = MockLLMProvider(
        responses=[
            # Supervisor routes to data agent
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            # Data agent calculates
            LLMResponse(content="Base calculation done: 1000.", model="mock-llm"),
            # Supervisor routes to final
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            # Final agent synthesizes answer with modified parameters
            LLMResponse(
                content="Please note always use discount factors. Final result: 900.",
                model="mock-llm",
            ),
        ]
    )

    thread_id = "phase6_modify_thread"
    response = await run_orchestrated_task(
        task="Calculate standard price",
        settings=settings,
        thread_id=thread_id,
        scope_id="user_mod_1",
        require_human_review=True,
        provider=mock_llm,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )

    # 1. Workflow pauses on interrupt
    assert response.status == "interrupted"

    # 2. Verify NO memory extraction while task is paused
    count_before_resume = await mem_store.count(scope_id="user_mod_1")
    assert count_before_resume == 0

    # 3. Resume with MODIFY
    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )
    hitl_service = HITLService(checkpointer=checkpointer, settings=settings)
    final_res = await hitl_service.resume_thread(
        thread_id=thread_id,
        response=HITLResponse(
            decision=ApprovalDecision.MODIFY,
            feedback="Apply 10% holiday discount",
            modified_action={
                "task": "Calculate standard price with 10% discount",
                "discount": 0.10,
            },
        ),
        graph=graph,
    )

    # 4. Verify completed state and parameters
    assert final_res["status"] == "completed"
    assert final_res["metadata"]["human_decision"] == "modified"
    assert final_res["metadata"]["modified_action"]["discount"] == 0.10

    # 5. Verify memory extracted after successful completion
    count_after_completion = await mem_store.count(scope_id="user_mod_1")
    assert count_after_completion >= 1


@pytest.mark.asyncio
async def test_orchestration_hitl_reject_no_memory_extraction():
    """Verify rejected workflows do not extract or persist memories."""
    settings = Settings(
        memory_enabled=True,
        hitl_enabled=True,
        checkpoint_backend="memory",
    )
    checkpointer = MemorySaver()
    embedder = MockEmbeddingProvider(dimension=16)

    mem_store = ChromaMemoryStore(
        collection_name="test_integ_reject_mem",
        embedding_provider=embedder,
        ephemeral=True,
    )
    memory_service = MemoryService(store=mem_store, settings=settings)

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Preliminary plan generated.", model="mock-llm"),
        ]
    )

    thread_id = "phase6_reject_thread"
    response = await run_orchestrated_task(
        task="Execute unauthorized privileged operation",
        settings=settings,
        thread_id=thread_id,
        scope_id="user_rej_1",
        require_human_review=True,
        provider=mock_llm,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )

    assert response.status == "interrupted"
    assert await mem_store.count(scope_id="user_rej_1") == 0

    # Resume with REJECT
    graph = build_orchestration_graph(
        provider=mock_llm,
        checkpointer=checkpointer,
        memory_service=memory_service,
    )
    hitl_service = HITLService(checkpointer=checkpointer, settings=settings)
    final_res = await hitl_service.resume_thread(
        thread_id=thread_id,
        response=HITLResponse(
            decision=ApprovalDecision.REJECT,
            feedback="Blocked by security auditor.",
        ),
        graph=graph,
    )

    assert final_res["status"] == "error"
    assert "aborted by human reviewer" in final_res["final_answer"]

    # Verify strictly 0 memories extracted after rejection
    count_after_reject = await mem_store.count(scope_id="user_rej_1")
    assert count_after_reject == 0
