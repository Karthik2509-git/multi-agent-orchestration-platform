"""Phase 7 Milestone 1: OpenTelemetry Core, Tracing Spans & Telemetry Redaction tests."""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings
from src.app.hitl.models import ApprovalDecision, HITLResponse
from src.app.hitl.service import HITLService
from src.app.llm.providers.mock import MockLLMProvider
from src.app.mcp.adapters import MCPToolAdapter
from src.app.mcp.client import MCPClient
from src.app.mcp.models import MCPToolDefinition
from src.app.mcp.servers.local_tools import create_local_mcp_server
from src.app.memory.models import MemoryType
from src.app.memory.service import MemoryService
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore
from src.app.models.schemas.llm import LLMResponse
from src.app.observability import (
    get_telemetry_manager,
    get_tracer,
    init_telemetry,
    trace_span,
)
from src.app.observability.redaction import (
    clean_text_length,
    sanitize_attributes,
    sanitize_error,
)
from src.app.orchestration.graph import build_orchestration_graph
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService
from src.app.services.orchestration_service import run_orchestrated_task
from src.app.tools.calculator import CalculatorTool
from src.app.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clean_telemetry_state():
    """Ensure telemetry starts clean and resets after each test."""
    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        telemetry_service_name="multi-agent-orchestrator-test",
        telemetry_record_payloads=False,
        telemetry_max_in_memory_spans=500,
        checkpoint_backend="memory",
    )
    manager = init_telemetry(settings)
    manager.clear_in_memory_spans()
    yield manager
    manager.clear_in_memory_spans()


# -----------------------------------------------------------------------------
# 1. Telemetry initializes when enabled
# -----------------------------------------------------------------------------
def test_telemetry_initializes_when_enabled():
    """Verify telemetry initializes properly when enabled in settings."""
    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        telemetry_service_name="orchestrator-init-test",
        checkpoint_backend="memory",
    )
    manager = init_telemetry(settings)
    assert manager.is_initialized() is True
    assert manager.is_enabled is True
    assert manager.service_name == "orchestrator-init-test"
    assert manager.get_tracer_provider() is not None

    tracer = get_tracer("test_tracer")
    assert tracer is not None


# -----------------------------------------------------------------------------
# 2. Telemetry can be disabled
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telemetry_can_be_disabled():
    """Verify telemetry can be disabled without causing application errors."""
    settings = Settings(
        telemetry_enabled=False,
        telemetry_exporter="none",
        checkpoint_backend="memory",
    )
    manager = init_telemetry(settings)
    assert manager.is_initialized() is False
    assert manager.is_enabled is False

    # trace_span should be a no-op that yields None without error
    async with trace_span("disabled.span", attributes={"test": 1}) as span:
        assert span is None

    assert len(manager.get_in_memory_spans()) == 0


# -----------------------------------------------------------------------------
# 3. In-memory exporter collects spans
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_in_memory_exporter_collects_spans():
    """Verify BoundedInMemorySpanExporter buffers spans with attributes."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    async with trace_span("sample.operation", attributes={"batch_size": 10, "mode": "fast"}):
        pass

    spans = manager.get_in_memory_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "sample.operation"
    assert span.attributes.get("batch_size") == 10
    assert span.attributes.get("mode") == "fast"


# -----------------------------------------------------------------------------
# 4. Production forces telemetry_record_payloads=False
# -----------------------------------------------------------------------------
def test_production_forces_telemetry_record_payloads_false():
    """Verify that app_env='production' strictly overrides telemetry_record_payloads to False."""
    settings = Settings(
        app_env="production",
        telemetry_record_payloads=True,  # Attempting to enable in prod
        checkpoint_backend="memory",
    )
    assert settings.telemetry_record_payloads is False


# -----------------------------------------------------------------------------
# 5. Orchestration creates orchestration.run span
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestration_run_span_created():
    """Verify run_orchestrated_task emits an orchestration.run span with expected metadata."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final synthesized output.", model="mock-llm"),
        ]
    )

    result = await run_orchestrated_task(
        task="Test run task",
        settings=settings,
        thread_id="test_thread_orch_span",
        scope_id="user_scope_42",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )
    assert result.status == "completed"

    spans = manager.get_in_memory_spans()
    orch_spans = [s for s in spans if s.name == "orchestration.run"]
    assert len(orch_spans) == 1
    orch_span = orch_spans[0]

    assert orch_span.attributes.get("thread_id") == "test_thread_orch_span"
    assert orch_span.attributes.get("scope_id") == "user_scope_42"
    assert orch_span.attributes.get("status") in ("completed", "success")
    assert orch_span.attributes.get("task_chars") == len("Test run task")
    assert "execution_time_seconds" in orch_span.attributes


# -----------------------------------------------------------------------------
# 6. Supervisor span exists
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_supervisor_span_exists():
    """Verify supervisor routing creates supervisor.decide_route span."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Done.", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Routing test task",
        settings=settings,
        thread_id="test_thread_supervisor_span",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    sup_spans = [s for s in spans if s.name == "supervisor.decide_route"]
    assert len(sup_spans) >= 1
    sup_span = sup_spans[0]
    assert sup_span.attributes.get("status") in ("completed", "success")
    assert sup_span.attributes.get("selected_route") == "final"
    assert sup_span.attributes.get("allowed_route") is True
    assert sup_span.attributes.get("step_count") == 0


# -----------------------------------------------------------------------------
# 7. Agent span exists
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_span_exists():
    """Verify agent execution creates agent.execute span with agent_name."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            LLMResponse(content="Data agent finished analysis.", model="mock-llm"),
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final summary.", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Agent test task",
        settings=settings,
        thread_id="test_thread_agent_span",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    agent_spans = [s for s in spans if s.name == "agent.execute"]
    assert len(agent_spans) >= 2  # data agent and final agent
    data_agent_span = next(s for s in agent_spans if s.attributes.get("agent_name") == "data")
    assert data_agent_span.attributes.get("status") in ("completed", "success")
    assert data_agent_span.attributes.get("step_number") == 1


# -----------------------------------------------------------------------------
# 8. Tool span exists when tool executes
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_execute_span_for_native_tool():
    """Verify native tool execution produces tool.execute span."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    result = await registry.execute("calculator", {"expression": "42 * 2"})
    assert result.success is True

    spans = manager.get_in_memory_spans()
    tool_spans = [s for s in spans if s.name == "tool.execute"]
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span.attributes.get("tool_name") == "calculator"
    assert tool_span.attributes.get("tool_type") == "native"
    assert tool_span.attributes.get("status") in ("completed", "success")
    assert "duration_ms" in tool_span.attributes


# -----------------------------------------------------------------------------
# 9. MCP span metadata exists for MCP execution
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mcp_tool_execute_span_with_metadata():
    """Verify MCP tool execution produces tool.execute span with server_name and original_name."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    server = create_local_mcp_server(name="local_mcp_test")
    client = MCPClient(server_target=server, server_name="local_mcp_test")

    defn = MCPToolDefinition(
        name="mcp.local_mcp_test.calculator",
        server_name="local_mcp_test",
        original_name="calculator",
        description="Safe arithmetic evaluation",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )

    adapter = MCPToolAdapter(definition=defn, client=client)
    res = await adapter.execute(expression="10 + 20")
    assert res.success is True

    await client.close()

    spans = manager.get_in_memory_spans()
    tool_spans = [s for s in spans if s.name == "tool.execute"]
    assert len(tool_spans) == 1
    mcp_span = tool_spans[0]
    assert mcp_span.attributes.get("tool_name") == "mcp.local_mcp_test.calculator"
    assert mcp_span.attributes.get("tool_type") == "mcp"
    assert mcp_span.attributes.get("mcp.server_name") == "local_mcp_test"
    assert mcp_span.attributes.get("mcp.original_name") == "calculator"
    assert mcp_span.attributes.get("status") in ("completed", "success")


# -----------------------------------------------------------------------------
# 10. Memory retrieval span exists
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_memory_retrieve_span_created():
    """Verify memory retrieval creates memory.retrieve span."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        memory_enabled=True,
        checkpoint_backend="memory",
    )
    embedder = MockEmbeddingProvider(dimension=16)
    mem_store = ChromaMemoryStore(
        collection_name="test_obs_mem",
        embedding_provider=embedder,
        ephemeral=True,
    )
    memory_service = MemoryService(store=mem_store, settings=settings)

    await memory_service.add_memory(
        content="User preference: Always respond concisely.",
        scope_id="user_obs_1",
        memory_type=MemoryType.USER_PREFERENCE,
    )

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final concise answer.", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Provide concise report",
        settings=settings,
        thread_id="test_thread_mem_span",
        scope_id="user_obs_1",
        provider=mock_llm,
        checkpointer=MemorySaver(),
        memory_service=memory_service,
    )

    spans = manager.get_in_memory_spans()
    mem_spans = [s for s in spans if s.name == "memory.retrieve"]
    assert len(mem_spans) == 1
    mem_span = mem_spans[0]
    assert mem_span.attributes.get("scope_id") == "user_obs_1"
    assert mem_span.attributes.get("hit_count") >= 1
    assert mem_span.attributes.get("status") in ("completed", "success")


# -----------------------------------------------------------------------------
# 11. RAG retrieval span exists
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rag_retrieve_span_created():
    """Verify RAGService.retrieve creates rag.retrieve span with retrieval metrics."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        rag_enabled=True,
        rag_embedding_provider="mock",
        rag_persist_directory=None,
        checkpoint_backend="memory",
    )
    vector_store = ChromaVectorStore(
        collection_name="test_obs_rag",
        ephemeral=True,
    )
    embedder = MockEmbeddingProvider(dimension=16)
    rag_service = RAGService(
        vector_store=vector_store,
        embedding_provider=embedder,
        settings=settings,
    )

    await rag_service.ingest_text(
        content="OpenTelemetry provides standardized APIs and SDKs for distributed tracing.",
        filename="otel_overview.txt",
    )

    results = await rag_service.retrieve(query="distributed tracing standard", top_k=2)
    assert len(results) >= 1

    spans = manager.get_in_memory_spans()
    rag_spans = [s for s in spans if s.name == "rag.retrieve"]
    assert len(rag_spans) == 1
    rag_span = rag_spans[0]
    assert rag_span.attributes.get("strategy") == "hybrid"
    assert rag_span.attributes.get("top_k") == 2
    assert rag_span.attributes.get("result_count") >= 1
    assert rag_span.attributes.get("status") in ("completed", "success")
    assert "duration_ms" in rag_span.attributes


# -----------------------------------------------------------------------------
# 12. HITL interrupt and resume spans exist
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hitl_interrupt_and_resume_spans():
    """Verify HITL interruption produces hitl.interrupt span and resumption produces hitl.resume."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        hitl_enabled=True,
        checkpoint_backend="memory",
    )
    checkpointer = MemorySaver()
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            LLMResponse(content="Data result computed.", model="mock-llm"),
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final approved response.", model="mock-llm"),
        ]
    )

    thread_id = "test_thread_hitl_spans"
    res = await run_orchestrated_task(
        task="Action requiring review",
        settings=settings,
        thread_id=thread_id,
        require_human_review=True,
        provider=mock_llm,
        checkpointer=checkpointer,
    )
    assert res.status == "interrupted"

    spans = manager.get_in_memory_spans()
    interrupt_spans = [s for s in spans if s.name == "hitl.interrupt"]
    assert len(interrupt_spans) == 1
    int_span = interrupt_spans[0]
    assert int_span.attributes.get("status") == "interrupted"
    assert int_span.attributes.get("approval_level") in ("approve_plan", "task_level")

    # Now resume thread
    graph = build_orchestration_graph(provider=mock_llm, checkpointer=checkpointer)
    hitl_service = HITLService(checkpointer=checkpointer, settings=settings)

    manager.clear_in_memory_spans()
    resume_res = await hitl_service.resume_thread(
        thread_id=thread_id,
        response=HITLResponse(decision=ApprovalDecision.APPROVE),
        graph=graph,
    )
    assert resume_res["status"] == "completed"

    spans_after_resume = manager.get_in_memory_spans()
    resume_spans = [s for s in spans_after_resume if s.name == "hitl.resume"]
    assert len(resume_spans) == 1
    res_span = resume_spans[0]
    assert res_span.attributes.get("thread_id") == thread_id
    assert res_span.attributes.get("decision") == "approve"
    assert res_span.attributes.get("status") == "resumed"


# -----------------------------------------------------------------------------
# 13. Final synthesis span exists
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_final_synthesis_span_created():
    """Verify final synthesis produces final.synthesis span."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final synthesized output answer.", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Synthesize response",
        settings=settings,
        thread_id="test_thread_synth_span",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    synth_spans = [s for s in spans if s.name == "final.synthesis"]
    assert len(synth_spans) == 1
    synth_span = synth_spans[0]
    assert synth_span.attributes.get("status") in ("completed", "success")
    answer_len = synth_span.attributes.get("answer_chars") or synth_span.attributes.get(
        "synthesis_chars"
    )
    assert answer_len == len("Final synthesized output answer.")


# -----------------------------------------------------------------------------
# 14. Parent/child trace relationships are correct
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_trace_parent_child_hierarchy():
    """Verify proper parent/child hierarchy:
    orchestration.run is ancestor of supervisor, agent, and final synthesis.
    """
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            LLMResponse(content="Data result 42", model="mock-llm"),
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final synthesized 42", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Hierarchy test task",
        settings=settings,
        thread_id="test_thread_hierarchy",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    span_by_name: Dict[str, List[Any]] = {}
    for s in spans:
        span_by_name.setdefault(s.name, []).append(s)

    orch_span = span_by_name["orchestration.run"][0]
    orch_span_id = orch_span.context.span_id

    # Root span has no parent
    assert orch_span.parent is None or orch_span.parent.span_id is None

    # Supervisor span parent is orchestration.run
    supervisor_span = span_by_name["supervisor.decide_route"][0]
    assert supervisor_span.parent is not None
    assert supervisor_span.parent.span_id == orch_span_id

    # Agent span parent is orchestration.run
    data_agent_span = [
        s for s in span_by_name["agent.execute"] if s.attributes.get("agent_name") == "data"
    ][0]
    assert data_agent_span.parent is not None
    assert data_agent_span.parent.span_id == orch_span_id

    # Final agent execute parent is orchestration.run
    final_agent_span = [
        s for s in span_by_name["agent.execute"] if s.attributes.get("agent_name") == "final"
    ][0]
    assert final_agent_span.parent is not None
    assert final_agent_span.parent.span_id == orch_span_id

    # Final synthesis span parent is final agent.execute
    final_synth_span = span_by_name["final.synthesis"][0]
    assert final_synth_span.parent is not None
    assert final_synth_span.parent.span_id == final_agent_span.context.span_id


# -----------------------------------------------------------------------------
# 15. All relevant spans share the same trace_id
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_spans_share_same_trace_id():
    """Verify that all spans generated within an orchestrated task share the exact same trace_id."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}', model="mock-llm"),
            LLMResponse(content="Data answer", model="mock-llm"),
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final complete output", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Shared trace ID test",
        settings=settings,
        thread_id="test_thread_trace_id",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    assert len(spans) > 5

    orch_span = next(s for s in spans if s.name == "orchestration.run")
    expected_trace_id = orch_span.context.trace_id

    for s in spans:
        assert s.context.trace_id == expected_trace_id, (
            f"Span {s.name} has trace_id {s.context.trace_id}, expected {expected_trace_id}"
        )


# -----------------------------------------------------------------------------
# 16. Raw task text is absent from span attributes
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_raw_task_absent_from_span_attributes():
    """Verify that sensitive user prompt / raw task text is never stored in span attributes."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    secret_task_text = "CONFIDENTIAL_TASK_QUERY_998877665544"
    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        telemetry_record_payloads=False,
        checkpoint_backend="memory",
    )
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Task complete.", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task=secret_task_text,
        settings=settings,
        thread_id="test_thread_no_raw_task",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    for s in spans:
        assert "raw_task" not in s.attributes
        assert "task" not in s.attributes
        for attr_key, attr_val in s.attributes.items():
            if isinstance(attr_val, str):
                assert secret_task_text not in attr_val, (
                    f"Secret task leaked into span {s.name} attribute {attr_key}"
                )


# -----------------------------------------------------------------------------
# 17. Full LLM output is absent from span attributes
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_llm_output_absent_from_span_attributes():
    """Verify that full LLM generation text is not stored in span attributes."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    secret_llm_text = "TOP_SECRET_LLM_GENERATED_COMPLETION_12345"
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content=secret_llm_text, model="mock-llm"),
        ]
    )

    await mock_llm.generate(messages=[{"role": "user", "content": "Hello"}])

    spans = manager.get_in_memory_spans()
    llm_spans = [s for s in spans if s.name == "llm.call"]
    assert len(llm_spans) == 1
    llm_span = llm_spans[0]

    for attr_key, attr_val in llm_span.attributes.items():
        assert attr_key not in {"content", "completion", "generated_text", "response"}
        if isinstance(attr_val, str):
            assert secret_llm_text not in attr_val


# -----------------------------------------------------------------------------
# 18. RAG chunk contents are absent from span attributes
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rag_chunk_contents_absent_from_span_attributes():
    """Verify that chunk text retrieved from knowledge base is never stored in span attributes."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    secret_rag_content = "PROPRIETARY_INTERNAL_DOCUMENTATION_CHUNK_8899"
    settings = Settings(
        rag_enabled=True,
        rag_embedding_provider="mock",
        rag_persist_directory=None,
        checkpoint_backend="memory",
    )
    vector_store = ChromaVectorStore(
        collection_name="test_obs_rag_secret",
        ephemeral=True,
    )
    embedder = MockEmbeddingProvider(dimension=16)
    rag_service = RAGService(
        vector_store=vector_store,
        embedding_provider=embedder,
        settings=settings,
    )

    await rag_service.ingest_text(
        content=secret_rag_content,
        filename="secret.txt",
    )

    await rag_service.retrieve(query="INTERNAL", top_k=1)

    spans = manager.get_in_memory_spans()
    rag_spans = [s for s in spans if s.name == "rag.retrieve"]
    assert len(rag_spans) == 1
    rag_span = rag_spans[0]

    for attr_key, attr_val in rag_span.attributes.items():
        assert attr_key not in {"content", "chunks", "documents", "query"}
        if isinstance(attr_val, str):
            assert secret_rag_content not in attr_val


# -----------------------------------------------------------------------------
# 19. Memory contents are absent from span attributes
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_memory_contents_absent_from_span_attributes():
    """Verify that memory contents retrieved are never stored in span attributes."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    secret_memory = "SECRET_USER_PERSONAL_DATA_SSN_9999"
    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        memory_enabled=True,
        checkpoint_backend="memory",
    )
    embedder = MockEmbeddingProvider(dimension=16)
    mem_store = ChromaMemoryStore(
        collection_name="test_obs_mem_secret",
        embedding_provider=embedder,
        ephemeral=True,
    )
    memory_service = MemoryService(store=mem_store, settings=settings)

    await memory_service.add_memory(
        content=secret_memory,
        scope_id="user_secret_scope",
        memory_type=MemoryType.USER_PREFERENCE,
    )

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final answer.", model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task="Test memory redaction",
        settings=settings,
        thread_id="test_thread_mem_redact",
        scope_id="user_secret_scope",
        provider=mock_llm,
        checkpointer=MemorySaver(),
        memory_service=memory_service,
    )

    spans = manager.get_in_memory_spans()
    mem_spans = [s for s in spans if s.name == "memory.retrieve"]
    assert len(mem_spans) == 1
    mem_span = mem_spans[0]

    for attr_key, attr_val in mem_span.attributes.items():
        assert attr_key not in {"content", "memories", "text"}
        if isinstance(attr_val, str):
            assert secret_memory not in attr_val


# -----------------------------------------------------------------------------
# 20. Credentials and authorization headers are not recorded
# -----------------------------------------------------------------------------
def test_credentials_and_headers_redacted():
    """Verify sanitize_attributes purges credentials and converts payload strings safely."""
    raw_attrs = {
        "authorization": "Bearer eyJhbGciOi...",
        "api_key": "sk-1234567890abcdef",
        "secret_token": "secret_abc",
        "password": "my_password_123",
        "cookie": "session_id=abcdef",
        "raw_task": "Do this confidential task now",
        "prompt": "Here is the raw prompt",
        "completion": "Here is the generated output",
        "safe_int": 42,
        "safe_str": "normal_metadata",
        "safe_bool": True,
    }

    sanitized = sanitize_attributes(raw_attrs, record_payloads=False)

    # Sensitive keys must be completely removed
    assert "authorization" not in sanitized
    assert "api_key" not in sanitized
    assert "secret_token" not in sanitized
    assert "password" not in sanitized
    assert "cookie" not in sanitized

    # Payload keys should be converted to length measurements, not text
    assert "raw_task" not in sanitized
    assert "prompt" not in sanitized
    assert "completion" not in sanitized
    assert sanitized["raw_task_chars"] == len("Do this confidential task now")
    assert sanitized["prompt_chars"] == len("Here is the raw prompt")
    assert sanitized["completion_chars"] == len("Here is the generated output")

    # Safe attributes are preserved
    assert sanitized["safe_int"] == 42
    assert sanitized["safe_str"] == "normal_metadata"
    assert sanitized["safe_bool"] is True


def test_sanitize_error_strips_sensitive_information():
    """Verify sanitize_error produces safe error summaries."""
    err = RuntimeError(
        "Failed connecting with api_key=sk-secret-key-12345 at https://api.openai.com"
    )
    cleaned = sanitize_error(err)
    assert "api_key=" not in cleaned["error.message"]
    assert "sk-secret-key-12345" not in cleaned["error.message"]
    assert cleaned["error.type"] == "RuntimeError"


def test_clean_text_length_helper():
    """Verify clean_text_length helper converts strings to character length."""
    assert clean_text_length("hello world") == 11
    assert clean_text_length("") == 0
    assert clean_text_length(None) == 0


# -----------------------------------------------------------------------------
# 21. FastAPI instrumentation root span
# -----------------------------------------------------------------------------
def test_fastapi_instrumentation_root_span(client):
    """Verify FastAPI instrumentor traces HTTP requests without error."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    response = client.get("/health")
    assert response.status_code == 200

    spans = manager.get_in_memory_spans()
    # If spans captured by FastAPIInstrumentor, verify attributes;
    # regardless, health request succeeded
    if spans:
        http_span = spans[-1]
        assert (
            "http" in http_span.name.lower()
            or "health" in str(http_span.attributes)
            or "/" in http_span.name
        )


# -----------------------------------------------------------------------------
# 22. REAL HTTP -> orchestration trace hierarchy verification
# -----------------------------------------------------------------------------
def test_real_http_to_orchestration_trace_hierarchy(client):
    """Verify real HTTP POST /api/v1/orchestration/run produces connected trace hierarchy."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    mock_provider = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}'),
            LLMResponse(content="Data calculated at 42 units."),
            LLMResponse(content='{"next_agent": "final"}'),
            LLMResponse(content="Final synthesized output: 42 units."),
        ]
    )

    with patch(
        "src.app.services.orchestration_service.get_llm_provider",
        return_value=mock_provider,
    ):
        response = client.post(
            "/api/v1/orchestration/run",
            json={"task": "Compute statistical metrics"},
        )
        assert response.status_code == 200
        assert "42 units" in response.json()["answer"]

    spans = manager.get_in_memory_spans()
    assert len(spans) > 5

    # Find HTTP server span and orchestration.run span
    http_root_spans = [s for s in spans if s.parent is None]
    assert len(http_root_spans) == 1, "Exactly one root HTTP span must exist"
    http_span = http_root_spans[0]

    orch_spans = [s for s in spans if s.name == "orchestration.run"]
    assert len(orch_spans) == 1, "Exactly one orchestration.run span must exist"
    orch_span = orch_spans[0]

    # Verify same trace_id
    assert orch_span.context.trace_id == http_span.context.trace_id

    # Verify orchestration.run parent is the HTTP span
    assert orch_span.parent is not None
    assert orch_span.parent.span_id == http_span.context.span_id

    # Verify child spans descend from orchestration.run
    supervisor_spans = [s for s in spans if s.name == "supervisor.decide_route"]
    assert len(supervisor_spans) >= 1
    for s in supervisor_spans:
        assert s.parent is not None
        assert s.parent.span_id == orch_span.context.span_id
        assert s.context.trace_id == http_span.context.trace_id

    agent_spans = [s for s in spans if s.name == "agent.execute"]
    assert len(agent_spans) >= 2
    for s in agent_spans:
        assert s.parent is not None
        assert s.parent.span_id == orch_span.context.span_id
        assert s.context.trace_id == http_span.context.trace_id

    final_synth_spans = [s for s in spans if s.name == "final.synthesis"]
    assert len(final_synth_spans) == 1
    final_synth = final_synth_spans[0]
    final_agent = next(s for s in agent_spans if s.attributes.get("agent_name") == "final")
    assert final_synth.parent is not None
    assert final_synth.parent.span_id == final_agent.context.span_id
    assert final_synth.context.trace_id == http_span.context.trace_id


# -----------------------------------------------------------------------------
# 23. Verify no duplicate root trace
# -----------------------------------------------------------------------------
def test_no_duplicate_root_trace_on_http_orchestration(client):
    """Verify HTTP orchestration produces exactly 1 root span and
    orchestration.run is not a root.
    """
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    mock_provider = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}'),
            LLMResponse(content="Fast response."),
        ]
    )

    with patch(
        "src.app.services.orchestration_service.get_llm_provider",
        return_value=mock_provider,
    ):
        response = client.post(
            "/api/v1/orchestration/run",
            json={"task": "Root trace check"},
        )
        assert response.status_code == 200

    spans = manager.get_in_memory_spans()
    root_spans = [s for s in spans if s.parent is None]
    assert len(root_spans) == 1
    assert "orchestration.run" != root_spans[0].name

    orch_span = next(s for s in spans if s.name == "orchestration.run")
    assert orch_span.parent is not None
    assert orch_span.parent.span_id == root_spans[0].context.span_id


# -----------------------------------------------------------------------------
# 24. Verify production redaction through actual configuration
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_production_redaction_through_actual_configuration():
    """Verify app_env='production' enforces telemetry_record_payloads=False and purges secrets."""
    prod_settings = Settings(
        app_env="production",
        telemetry_enabled=True,
        telemetry_record_payloads=True,  # Attempting override in prod
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    # 1. Assert effective configuration forces False
    assert prod_settings.telemetry_record_payloads is False

    manager = init_telemetry(prod_settings)
    manager.clear_in_memory_spans()

    secret_raw_task = "VERY_CONFIDENTIAL_USER_QUERY_883311"
    secret_prompt = "INTERNAL_SYSTEM_PROMPT_SECRET_664422"
    secret_completion = "SENSITIVE_OUTPUT_DATA_115599"

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content=secret_completion, model="mock-llm"),
        ]
    )

    await run_orchestrated_task(
        task=secret_raw_task,
        settings=prod_settings,
        thread_id="prod_redact_thread_1",
        provider=mock_llm,
        checkpointer=MemorySaver(),
    )

    spans = manager.get_in_memory_spans()
    assert len(spans) >= 3

    forbidden_payload_keys = {"raw_task", "prompt", "completion", "messages", "content"}
    forbidden_credential_keys = {"authorization", "password", "api_key", "token", "secret"}

    for span in spans:
        for attr_key, attr_val in span.attributes.items():
            attr_lower = attr_key.lower()
            assert attr_lower not in forbidden_payload_keys, (
                f"Forbidden payload key '{attr_key}' in span '{span.name}'"
            )
            assert attr_lower not in forbidden_credential_keys, (
                f"Forbidden credential key '{attr_key}' in span '{span.name}'"
            )
            if isinstance(attr_val, str):
                for secret in [secret_raw_task, secret_prompt, secret_completion]:
                    assert secret not in attr_val, (
                        f"Secret leaked in span '{span.name}' attr '{attr_key}'"
                    )


# -----------------------------------------------------------------------------
# 25. Verify telemetry does not affect business behavior
# -----------------------------------------------------------------------------
def test_telemetry_does_not_affect_business_behavior(client):
    """Verify orchestration output is identical whether telemetry is enabled or disabled."""
    responses_proto = [
        LLMResponse(content='{"next_agent": "data"}'),
        LLMResponse(content="Analysis: Result value is 999."),
        LLMResponse(content='{"next_agent": "final"}'),
        LLMResponse(content="Final consolidated answer is 999."),
    ]

    # Run 1: Telemetry Enabled
    settings_enabled = Settings(
        app_env="testing",
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
    )
    init_telemetry(settings_enabled)
    mock_1 = MockLLMProvider(responses=[r.model_copy() for r in responses_proto])
    with patch("src.app.services.orchestration_service.get_llm_provider", return_value=mock_1):
        res_enabled = client.post(
            "/api/v1/orchestration/run",
            json={"task": "Check business invariance"},
        )
    assert res_enabled.status_code == 200
    data_enabled = res_enabled.json()

    # Run 2: Telemetry Disabled
    settings_disabled = Settings(
        app_env="testing",
        telemetry_enabled=False,
        telemetry_exporter="none",
        checkpoint_backend="memory",
    )
    init_telemetry(settings_disabled)
    mock_2 = MockLLMProvider(responses=[r.model_copy() for r in responses_proto])
    with patch("src.app.services.orchestration_service.get_llm_provider", return_value=mock_2):
        res_disabled = client.post(
            "/api/v1/orchestration/run",
            json={"task": "Check business invariance"},
        )
    assert res_disabled.status_code == 200
    data_disabled = res_disabled.json()

    # Assert business results are completely invariant
    assert data_enabled["status"] == data_disabled["status"] == "completed"
    assert data_enabled["answer"] == data_disabled["answer"]
    assert data_enabled["agents_used"] == data_disabled["agents_used"]
    assert data_enabled["task"] == data_disabled["task"]
