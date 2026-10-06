"""Phase 7 Milestone 6: Final APIs and Comprehensive End-to-End Integration Scenarios.

This suite exercises the full system across all major capabilities:
1. Multi-Agent Orchestration (Supervisor -> Specialist -> Final Synthesis)
2. Tool-Using Agent with Native Tools & Budget Tracking
3. MCP Tool Ecosystem Integration with Allowlisting & Guardrails
4. RAG & Knowledge Ingestion, Hybrid Search & Orchestration Integration
5. Long-Term Semantic Memory, Scope Isolation & Context Retrieval
6. Human-in-the-Loop (HITL) Interruption, Pending Inspection & Resume Decisions
7. Developer Execution Fork / Replay with Checkpoint Isolation
8. Deterministic Offline Evaluation Framework (Retrieval Metrics)
9. Observability Spans, Trace Propagation & Telemetry Redaction
10. Execution Controls: Tool Budgets, SSRF Protection & Failure Isolation
"""

from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings, get_settings
from src.app.evaluation import (
    calculate_context_assertion_overlap,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_retrieval_metrics,
    get_deterministic_benchmark_dataset,
)
from src.app.llm.providers.mock import MockLLMProvider
from src.app.main import create_app
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.observability import get_telemetry_manager
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService, get_rag_service
from src.app.replay.models import MockToolResult, ReplayModification
from src.app.replay.service import ReplayService
from src.app.services.agent_service import build_default_tool_registry
from src.app.services.mcp_service import MCPService, get_mcp_service
from src.app.services.orchestration_service import run_orchestrated_task
from src.app.tools.calculator import CalculatorTool
from src.app.tools.execution_context import ToolExecutionContext
from src.app.tools.guardrails import ToolGuardrails
from src.app.tools.http_tool import SafeHTTPGetTool
from src.app.tools.knowledge_search import KnowledgeSearchTool
from src.app.tools.registry import ToolRegistry

# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def test_app():
    """Create a configured test application with memory backend and mock embeddings."""
    app = create_app()
    test_settings = Settings(
        app_env="testing",
        debug=True,
        checkpoint_backend="memory",
        rag_embedding_provider="mock",
        rag_persist_directory=None,
        memory_persist_directory=None,
        memory_similarity_threshold=0.0,
        telemetry_enabled=True,
        telemetry_exporter="memory",
        telemetry_record_payloads=False,
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    return app


# =============================================================================
# E2E 1 — Basic Multi-Agent Execution
# =============================================================================


def test_e2e_01_basic_multi_agent_execution(test_app):
    """Verify HTTP request -> orchestration -> supervisor -> specialist -> final synthesis."""
    mock_llm = MockLLMProvider(
        responses=[
            # Supervisor routes to research specialist
            LLMResponse(
                content='{"next_agent": "research", "reasoning": "Investigate technology trends"}'
            ),
            # Research specialist executes
            LLMResponse(content="Multi-agent orchestration improves resilience by 40%."),
            # Supervisor routes to final synthesis
            LLMResponse(content='{"next_agent": "final", "reasoning": "Research is sufficient"}'),
            # Final agent synthesizes answer
            LLMResponse(
                content="Synthesized conclusion: Multi-agent systems improve resilience by 40%."
            ),
        ]
    )

    with patch("src.app.services.orchestration_service.get_llm_provider", return_value=mock_llm):
        with TestClient(test_app) as client:
            response = client.post(
                "/api/v1/orchestration/run",
                json={"task": "Evaluate multi-agent resilience metrics"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "completed"
            assert "resilience by 40%" in data["answer"]
            assert "research" in data["agents_used"]
            assert "final" in data["agents_used"]
            assert data["execution_time_seconds"] >= 0.0


# =============================================================================
# E2E 2 — Tool-Using Agent
# =============================================================================


def test_e2e_02_tool_using_agent(test_app):
    """Verify ToolCallingAgent executes native calculator tool and tracks tool budgets."""
    mock_llm = MockLLMProvider(
        responses=[
            # Agent calls calculator tool
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_calc_e2e",
                        name="calculator",
                        arguments={"expression": "(250 * 4) + 150"},
                    )
                ],
            ),
            # Agent synthesizes final answer with result
            LLMResponse(
                content="The calculated total computation value is 1150.",
                tool_calls=[],
                finish_reason="stop",
            ),
        ]
    )

    with patch("src.app.services.agent_service.get_llm_provider", return_value=mock_llm):
        with TestClient(test_app) as client:
            response = client.post(
                "/api/v1/agent/run",
                json={"task": "Compute (250 * 4) + 150 for capacity planning"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "completed"
            assert "1150" in data["answer"]
            assert len(data["tool_calls"]) == 1
            t_call = data["tool_calls"][0]
            assert t_call["tool_name"] == "calculator"
            assert t_call["success"] is True
            assert "1150" in str(t_call["result"])


# =============================================================================
# E2E 3 — MCP Tool Execution
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_03_mcp_tool_execution():
    """Verify MCP subsystem discovery, namespaced tool execution, and registry guardrails."""
    settings = Settings(
        mcp_enabled=True,
        mcp_local_server_enabled=True,
        mcp_local_server_name="local",
        mcp_allowed_tools=["mcp.local.calculator"],
    )

    mcp_service = MCPService(settings=settings)
    await mcp_service.initialize()

    app = create_app()
    app.dependency_overrides[get_mcp_service] = lambda: mcp_service

    try:
        with TestClient(app) as client:
            # 1. Verify MCP health
            health_resp = client.get("/api/v1/mcp/health")
            assert health_resp.status_code == status.HTTP_200_OK
            health_data = health_resp.json()
            assert health_data["enabled"] is True
            assert health_data["servers"].get("local") == "connected"

            # 2. Verify MCP tool listing
            tools_resp = client.get("/api/v1/mcp/tools")
            assert tools_resp.status_code == status.HTTP_200_OK
            tools_data = tools_resp.json()
            assert len(tools_data["servers"]) >= 1
            server_tools = [t["name"] for t in tools_data["servers"][0]["tools"]]
            assert "mcp.local.calculator" in server_tools

        # 3. Verify ToolRegistry integration and execution through agent
        registry = build_default_tool_registry(settings=settings, mcp_service=mcp_service)
        assert registry.has_tool("mcp.local.calculator")

        mock_llm = MockLLMProvider(
            responses=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_mcp_e2e",
                            name="mcp.local.calculator",
                            arguments={"expression": "75 * 4"},
                        )
                    ],
                ),
                LLMResponse(content="Computed result via MCP server is 300."),
            ]
        )

        from src.app.agents.tool_calling_agent import ToolCallingAgent

        agent = ToolCallingAgent(provider=mock_llm, registry=registry, max_iterations=3)
        res = await agent.run("Calculate 75 * 4 using MCP")

        assert res.status == "completed"
        assert "300" in res.answer
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].tool_name == "mcp.local.calculator"
        assert res.tool_calls[0].success is True
    finally:
        await mcp_service.shutdown()


# =============================================================================
# E2E 4 — RAG Workflow
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_04_rag_workflow():
    """Verify document ingestion -> vector storage -> hybrid search -> citation generation."""
    settings = Settings(rag_enabled=True, rag_embedding_provider="mock", llm_provider="mock")
    store = ChromaVectorStore(collection_name="test_e2e_rag_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    rag_mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Project Hyperion launches on November 15, 2026 at Site B.",
                model="mock-llm",
            )
        ]
    )
    rag_service = RAGService(
        vector_store=store,
        embedding_provider=embedder,
        settings=settings,
        llm_provider=rag_mock_llm,
    )

    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    with TestClient(app) as client:
        # 1. Ingest document via API
        ingest_resp = client.post(
            "/api/v1/knowledge/documents",
            json={
                "content": (
                    "Project Hyperion is scheduled to launch on November 15, 2026 at Site B."
                ),
                "filename": "hyperion_specs.txt",
                "metadata": {"project": "Hyperion", "confidential": False},
            },
        )
        assert ingest_resp.status_code == status.HTTP_201_CREATED
        doc_id = ingest_resp.json()["document_id"]
        assert doc_id is not None

        # 2. Search knowledge base via API
        search_resp = client.post(
            "/api/v1/knowledge/search",
            json={"query": "When will Project Hyperion launch?", "top_k": 3},
        )
        assert search_resp.status_code == status.HTTP_200_OK
        search_data = search_resp.json()
        assert search_data["count"] >= 1
        assert "hyperion_specs.txt" in search_data["results"][0]["source"]
        assert "November 15, 2026" in search_data["results"][0]["content"]

        # 3. Grounded query with citations via API
        query_resp = client.post(
            "/api/v1/knowledge/query",
            json={"query": "Provide the launch date and location for Hyperion.", "top_k": 2},
        )
        assert query_resp.status_code == status.HTTP_200_OK
        query_data = query_resp.json()
        assert len(query_data["citations"]) >= 1
        assert query_data["citations"][0]["source"] == "hyperion_specs.txt"
        assert "November 15, 2026" in query_data["answer"]

    # 4. Integrate KnowledgeSearchTool in ToolRegistry for agent execution
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(rag_service=rag_service))
    assert registry.has_tool("knowledge_search")

    exec_result = await registry.execute("knowledge_search", {"query": "Hyperion launch site"})
    assert exec_result.success is True
    assert exec_result.data["count"] >= 1
    assert "Site B" in exec_result.data["results"][0]["content"]


# =============================================================================
# E2E 5 — Memory Workflow
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_05_memory_workflow(test_app):
    """Verify semantic memory creation -> scope isolation -> cross-scope segregation."""
    from src.app.memory.service import MemoryService
    from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore

    settings = Settings(memory_enabled=True, checkpoint_backend="memory")
    embedder = MockEmbeddingProvider(dimension=16)
    mem_store = ChromaMemoryStore(
        collection_name="test_e2e_mem_store",
        embedding_provider=embedder,
        ephemeral=True,
    )
    mem_service = MemoryService(store=mem_store, settings=settings)

    with patch("src.app.api.v1.endpoints.memory.get_memory_service", return_value=mem_service):
        with TestClient(test_app) as client:
            # 1. Create memory in scope "team_alpha"
            create_resp = client.post(
                "/api/v1/memory",
                json={
                    "content": (
                        "Coding convention: all models must inherit from Pydantic BaseModel."
                    ),
                    "scope_id": "team_alpha",
                    "memory_type": "user_preference",
                    "importance": 0.9,
                },
            )
            assert create_resp.status_code == status.HTTP_201_CREATED
            mem_id = create_resp.json()["memory"]["id"]

            # 2. Search memories in matching scope "team_alpha"
            search_alpha = client.post(
                "/api/v1/memory/search",
                json={
                    "query": "BaseModel inheritance rules",
                    "scope_id": "team_alpha",
                    "top_k": 3,
                },
            )
            assert search_alpha.status_code == status.HTTP_200_OK
            alpha_data = search_alpha.json()
            assert alpha_data["count"] >= 1
            assert "BaseModel" in alpha_data["results"][0]["memory"]["content"]

            # 3. Scope Isolation: search in different scope "team_beta" must return 0 results
            search_beta = client.post(
                "/api/v1/memory/search",
                json={"query": "BaseModel inheritance rules", "scope_id": "team_beta", "top_k": 3},
            )
            assert search_beta.status_code == status.HTTP_200_OK
            beta_data = search_beta.json()
            assert beta_data["count"] == 0, "Memory must NOT leak across scopes!"

            # 4. Verify memory retrieval endpoint
            get_resp = client.get(f"/api/v1/memory/{mem_id}?scope_id=team_alpha")
            assert get_resp.status_code == status.HTTP_200_OK
            assert get_resp.json()["memory"]["id"] == mem_id


# =============================================================================
# E2E 6 — HITL Workflow
# =============================================================================


def test_e2e_06_hitl_workflow(test_app):
    """Verify task pauses at approval gate -> inspect pending approval -> resume with decision."""
    checkpointer = MemorySaver()

    mock_llm = MockLLMProvider(
        responses=[
            # Step 1: Supervisor routes to final
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            # Step 2: Final agent synthesizes answer upon approval resume
            LLMResponse(content="Approved production action executed safely.", model="mock-llm"),
        ]
    )

    thread_id = "e2e_hitl_test_thread"

    with (
        patch("src.app.services.orchestration_service.get_llm_provider", return_value=mock_llm),
        patch("src.app.services.orchestration_service.get_checkpointer", return_value=checkpointer),
        patch("src.app.api.v1.endpoints.hitl.get_llm_provider", return_value=mock_llm),
        patch("src.app.api.v1.endpoints.hitl.get_checkpointer", return_value=checkpointer),
    ):
        with TestClient(test_app) as client:
            # 1. Trigger orchestration with human review required
            run_resp = client.post(
                "/api/v1/orchestration/run",
                json={
                    "task": "Deploy payment gateway migration",
                    "thread_id": thread_id,
                    "require_human_review": True,
                },
            )
            assert run_resp.status_code == status.HTTP_200_OK
            run_data = run_resp.json()
            assert run_data["status"] == "interrupted"
            assert run_data["pending_approval"] is not None

            # 2. Inspect pending approval via HITL API
            pending_resp = client.get(f"/api/v1/hitl/pending/{thread_id}")
            assert pending_resp.status_code == status.HTTP_200_OK
            pending_data = pending_resp.json()
            assert pending_data["has_pending_approval"] is True
            assert pending_data["request"]["thread_id"] == thread_id

            # 3. Resume with APPROVE
            resume_resp = client.post(
                f"/api/v1/hitl/resume/{thread_id}",
                json={"decision": "approve", "feedback": "Signed off by security engineer"},
            )
            assert resume_resp.status_code == status.HTTP_200_OK
            resume_data = resume_resp.json()
            assert resume_data["status"] == "completed"
            assert "Approved production action" in resume_data["answer"]


# =============================================================================
# E2E 7 — Developer Replay Workflow
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_07_replay_workflow(test_app):
    """Verify source execution is immutable and replay creates independent thread ID."""
    checkpointer = MemorySaver()
    settings = Settings(checkpoint_backend="memory")

    # 1. Run source execution
    source_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}'),
            LLMResponse(content="Initial computation: 100 units."),
            LLMResponse(content='{"next_agent": "final"}'),
            LLMResponse(content="Initial synthesized result: 100 units."),
        ]
    )
    source_thread = "e2e_source_thread_7"
    source_run = await run_orchestrated_task(
        task="Original baseline task",
        settings=settings,
        thread_id=source_thread,
        provider=source_llm,
        checkpointer=checkpointer,
    )
    assert source_run.status == "completed"

    # Capture source checkpoint state
    config_dict = {"configurable": {"thread_id": source_thread}}
    source_tuple_before = await checkpointer.aget_tuple(config_dict)
    source_step_before = source_tuple_before.checkpoint["channel_values"]["step_count"]

    # 2. Run execution replay with task and mock tool modification
    replay_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}'),
            LLMResponse(content="Replayed computation with mock tool: 9999 units."),
            LLMResponse(content='{"next_agent": "final"}'),
            LLMResponse(content="Replayed synthesized result: 9999 units."),
        ]
    )

    replay_service = ReplayService(settings=settings, checkpointer=checkpointer)
    modifications = ReplayModification(
        task_override="Forked task with mock tool override",
        mock_tool_results={"calculator": MockToolResult(success=True, data={"result": "9999"})},
    )

    replay_res = await replay_service.fork_and_replay(
        source_thread_id=source_thread,
        modifications=modifications,
        provider=replay_llm,
        checkpointer=checkpointer,
    )

    # 3. Assert replay isolation and correctness
    assert replay_res.success is True
    assert replay_res.source_thread_id == source_thread
    assert replay_res.replay_thread_id != source_thread
    assert "9999 units" in replay_res.answer

    # 4. Verify source checkpoint was completely unmodified
    source_tuple_after = await checkpointer.aget_tuple(config_dict)
    assert source_tuple_after.checkpoint["id"] == source_tuple_before.checkpoint["id"]
    assert source_tuple_after.checkpoint["channel_values"]["step_count"] == source_step_before
    assert source_tuple_after.checkpoint["channel_values"]["task"] == "Original baseline task"


# =============================================================================
# E2E 8 — Evaluation Workflow
# =============================================================================


def test_e2e_08_evaluation_workflow():
    """Verify deterministic evaluation framework metrics (Recall@K, Precision@K, MRR, overlap)."""
    dataset = get_deterministic_benchmark_dataset()
    assert len(dataset.cases) >= 3

    # Test individual metric mathematics
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    relevant = ["doc_2", "doc_5"]

    r_at_3 = calculate_recall_at_k(retrieved, relevant, k=3)
    p_at_3 = calculate_precision_at_k(retrieved, relevant, k=3)
    mrr_score, rank = calculate_reciprocal_rank(retrieved, relevant)

    assert r_at_3 == 0.5  # 1 out of 2 found in top 3
    assert p_at_3 == pytest.approx(1 / 3, rel=1e-3)
    assert mrr_score == 0.5  # first relevant item is at rank 2 -> 1/2
    assert rank == 2

    overlap = calculate_context_assertion_overlap(
        context="System requires AES-256 encryption at rest and TLS 1.3 in transit.",
        assertion="System uses AES-256 encryption and TLS 1.3.",
    )
    assert 0.0 <= overlap.context_assertion_overlap <= 1.0
    assert overlap.context_assertion_overlap > 0.4

    # Run retrieval metrics calculation
    metrics = calculate_retrieval_metrics(
        retrieved_chunk_ids=retrieved,
        relevant_chunk_ids=relevant,
        k=5,
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 0.4
    assert metrics.mrr == 0.5


# =============================================================================
# E2E 9 — Observability Workflow
# =============================================================================


def test_e2e_09_observability_workflow(test_app):
    """Verify trace propagation, root HTTP span hierarchy, and telemetry payload redaction."""
    manager = get_telemetry_manager()
    manager.clear_in_memory_spans()

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "data"}'),
            LLMResponse(content="Metrics calculation complete: 48."),
            LLMResponse(content='{"next_agent": "final"}'),
            LLMResponse(content="Final summary: 48."),
        ]
    )

    with patch("src.app.services.orchestration_service.get_llm_provider", return_value=mock_llm):
        with TestClient(test_app) as client:
            res = client.post(
                "/api/v1/orchestration/run",
                json={"task": "Secret task: sensitive user calculation"},
            )
            assert res.status_code == status.HTTP_200_OK

            spans = manager.get_in_memory_spans()
            assert len(spans) >= 4

            # 1. Exactly one root span
            root_spans = [s for s in spans if s.parent is None]
            assert len(root_spans) == 1, "There must be exactly one root span"
            http_root = root_spans[0]

            # 2. Child orchestration span connects to HTTP span
            orch_spans = [s for s in spans if s.name == "orchestration.run"]
            assert len(orch_spans) == 1
            orch_span = orch_spans[0]
            assert orch_span.context.trace_id == http_root.context.trace_id
            assert orch_span.parent.span_id == http_root.context.span_id

            # 3. Telemetry Redaction Audit: no sensitive raw prompts or tasks in attributes
            for span in spans:
                attrs = span.attributes or {}
                for k, v in attrs.items():
                    val_str = str(v)
                    assert "sensitive user calculation" not in val_str
                    assert "Secret task" not in val_str
                    assert k not in ("prompt", "completion", "payload")


# =============================================================================
# E2E 10 — Execution Controls & Security
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_10_execution_controls_and_budget():
    """Verify tool execution budgets, failure isolation, and SafeHTTPGetTool SSRF protection."""
    # 1. Tool execution budget enforcement
    guardrails = ToolGuardrails()
    context = ToolExecutionContext(max_tool_calls=2)
    registry = ToolRegistry(guardrails=guardrails)
    registry.register(CalculatorTool())

    # First 2 calls succeed
    call1 = await registry.execute("calculator", {"expression": "10 + 1"}, context=context)
    assert call1.success is True
    call2 = await registry.execute("calculator", {"expression": "20 + 2"}, context=context)
    assert call2.success is True

    # 3rd call exceeds max_calls budget
    call3 = await registry.execute("calculator", {"expression": "30 + 3"}, context=context)
    assert call3.success is False
    assert call3.error_category == "budget_exceeded"
    assert "budget exceeded" in call3.error.lower()

    # 2. SafeHTTPGetTool SSRF Protections
    http_tool = SafeHTTPGetTool(allowed_domains=["example.com", "api.example.com"])

    # Loopback IP attempt
    ssrf_loopback = await http_tool.execute(url="http://127.0.0.1:8000/secret")
    assert ssrf_loopback.success is False
    assert "not permitted" in ssrf_loopback.error.lower()

    # Cloud metadata IP attempt
    ssrf_metadata = await http_tool.execute(url="http://169.254.169.254/latest/meta-data")
    assert ssrf_metadata.success is False
    assert "not permitted" in ssrf_metadata.error.lower()

    # Non-allowlisted domain attempt
    ssrf_domain = await http_tool.execute(url="https://unauthorized-domain.com/data")
    assert ssrf_domain.success is False
    assert "not in the allowed domains" in ssrf_domain.error.lower()

    # 3. Tool Failure Isolation (unknown tool returns structured failure)
    unregistered = await registry.execute("unknown_tool", {}, context=context)
    assert unregistered.success is False
    assert unregistered.error_category in ("tool_not_found", "validation_error")


# =============================================================================
# API Error Contracts Verification
# =============================================================================


def test_api_error_contracts(test_app):
    """Verify structured, predictable error responses for invalid requests without stack traces."""
    with TestClient(test_app) as client:
        # 1. 422 for empty task
        r_empty = client.post("/api/v1/orchestration/run", json={"task": ""})
        assert r_empty.status_code == 422

        # 2. 404 for unknown replay thread
        r_replay_404 = client.post(
            "/api/v1/orchestration/replay",
            json={"source_thread_id": "non_existent_thread_xyz"},
        )
        assert r_replay_404.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in r_replay_404.json()["detail"].lower()

        # 3. 404 for non-existent memory ID
        r_mem_404 = client.get("/api/v1/memory/non_existent_memory_id")
        assert r_mem_404.status_code == status.HTTP_404_NOT_FOUND

        # 4. 404 for non-existent document ID
        r_doc_404 = client.delete("/api/v1/knowledge/documents/non_existent_doc_id")
        assert r_doc_404.status_code == status.HTTP_404_NOT_FOUND

        # 5. Health endpoints report clean status
        r_health = client.get("/api/v1/health")
        assert r_health.status_code == status.HTTP_200_OK
        assert r_health.json()["status"] == "healthy"
