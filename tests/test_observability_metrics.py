"""Comprehensive tests for Phase 7 Milestone 2 OpenTelemetry Metrics layer.

Verifies:
- Metric initialization and disabled no-op safety
- Strict low-cardinality constraints
- Domain metrics across Orchestration, Agents, LLM, Tools, RAG, Memory, and HITL
- Trace and metric coexistence
- Telemetry failure isolation
"""

import pytest

from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse
from src.app.observability.metrics import (
    ALLOWED_AGENT_ERROR_CATEGORIES,
    ALLOWED_HITL_REASONS,
    ALLOWED_ORCHESTRATION_ERROR_CATEGORIES,
    ALLOWED_TOOL_ERROR_CATEGORIES,
    categorize_agent_error,
    categorize_hitl_reason,
    categorize_orchestration_error,
    categorize_tool_error,
    record_agent_duration,
    record_agent_execution,
    record_agent_failure,
    record_hitl_approval_latency,
    record_hitl_decision,
    record_hitl_escalation,
    record_llm_metrics,
    record_memory_hits,
    record_memory_search,
    record_orchestration_duration,
    record_orchestration_failure,
    record_orchestration_request,
    record_rag_chunks_retrieved,
    record_rag_latency,
    record_rag_retrieval,
    record_tool_call,
    record_tool_failure,
    record_tool_latency,
)
from src.app.observability.telemetry import (
    get_in_memory_metrics,
    get_telemetry_manager,
    init_telemetry,
    reset_metrics,
    shutdown_telemetry,
)


@pytest.fixture(autouse=True)
def fresh_telemetry_state():
    """Ensure clean telemetry state for each test."""
    test_settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        telemetry_service_name="test-metric-service",
    )
    init_telemetry(test_settings)
    reset_metrics()
    yield
    shutdown_telemetry()


def get_metric_points(name: str):
    """Helper to extract data points for a given metric name from in-memory metrics."""
    mgr = get_telemetry_manager()
    return mgr.get_metric_data_points(name)


# -------------------------------------------------------------
# 1. Metric Creation & Disabled No-Op Safety
# -------------------------------------------------------------
def test_metrics_initialize_when_telemetry_enabled():
    """Test 1: Metrics initialize and record data points when telemetry is enabled."""
    mgr = get_telemetry_manager()
    assert mgr.is_enabled is True
    assert mgr.get_meter_provider() is not None

    record_orchestration_request("completed")
    pts = get_metric_points("orchestration_requests")
    assert len(pts) >= 1
    assert pts[-1].value == 1
    assert pts[-1].attributes.get("status") == "completed"


def test_metrics_safe_no_op_when_telemetry_disabled():
    """Test 2: Metrics become safe no-ops when telemetry is disabled."""
    disabled_settings = Settings(telemetry_enabled=False)
    init_telemetry(disabled_settings)
    mgr = get_telemetry_manager()
    assert mgr.is_enabled is False

    # Calling any metric function must safely succeed without throwing
    record_orchestration_request("completed")
    record_orchestration_failure("timeout")
    record_orchestration_duration(1.5, "completed")
    record_agent_execution("research", "success")
    record_agent_failure("research", "execution_error")
    record_agent_duration("research", 0.8)
    record_llm_metrics("mock", "model", 0.5, "success", 100, 50)
    record_tool_call("calculator", "native", "success")
    record_tool_failure("calculator", "timeout")
    record_tool_latency("calculator", "native", 0.05)
    record_rag_retrieval("hybrid", "success")
    record_rag_latency("hybrid", 0.2)
    record_rag_chunks_retrieved("hybrid", 4)
    record_memory_search("success")
    record_memory_hits(True)
    record_hitl_escalation("approve_action", "sensitive_action")
    record_hitl_decision("approve")
    record_hitl_approval_latency("approve", 5.0)

    # In-memory metrics should be None or empty
    assert get_in_memory_metrics() is None


# -------------------------------------------------------------
# 2. Strict Cardinality Constraints
# -------------------------------------------------------------
def test_strict_cardinality_no_high_cardinality_identifiers():
    """Tests 3-8: Metrics must NEVER contain thread_id, task_id, scope_id, trace_id,

    span_id, memory_id, document_id, chunk_id, request_id, or arbitrary user input.
    """
    record_orchestration_request("completed")
    record_orchestration_failure("provider_error")
    record_orchestration_duration(2.0, "completed")
    record_agent_execution("code", "success")
    record_agent_failure("code", "validation_error")
    record_agent_duration("code", 1.2)
    record_llm_metrics("mock", "mock-model", 0.3, "success", 200, 100)
    record_tool_call("calculator", "native", "success")
    record_tool_failure("calculator", "execution_error")
    record_tool_latency("calculator", "native", 0.02)
    record_rag_retrieval("hybrid", "success")
    record_rag_chunks_retrieved("hybrid", 5)
    record_rag_latency("hybrid", 0.1)
    record_memory_search("success")
    record_memory_hits(True)
    record_hitl_escalation("notify", "sensitive_action")
    record_hitl_decision("modify")
    record_hitl_approval_latency("modify", 3.0)

    metrics_data = get_in_memory_metrics()
    assert metrics_data is not None

    prohibited_keys = {
        "thread_id",
        "task_id",
        "scope_id",
        "trace_id",
        "span_id",
        "memory_id",
        "document_id",
        "chunk_id",
        "request_id",
        "user_id",
        "user",
        "prompt",
        "task",
        "error_message",
        "url",
    }

    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for pt in metric.data.data_points:
                    attrs = dict(pt.attributes)
                    for key in prohibited_keys:
                        assert key not in attrs, (
                            f"Prohibited key '{key}' found in metric '{metric.name}'"
                        )


def test_arbitrary_user_input_cannot_become_metric_dimension():
    """Test 8: Arbitrary user string passed to normalizers is coerced to bounded sets."""
    cat = categorize_orchestration_error("User says: 'Please transfer $1000000000 to account X'")
    assert cat in ALLOWED_ORCHESTRATION_ERROR_CATEGORIES

    agent_cat = categorize_agent_error("Arbitrary traceback with private email user@example.com")
    assert agent_cat in ALLOWED_AGENT_ERROR_CATEGORIES

    tool_cat = categorize_tool_error("https://secret-internal.corp/api/auth?token=xyz123")
    assert tool_cat in ALLOWED_TOOL_ERROR_CATEGORIES

    hitl_cat = categorize_hitl_reason("Arbitrary user prompt trying prompt injection")
    assert hitl_cat in ALLOWED_HITL_REASONS


# -------------------------------------------------------------
# 3. System / Orchestration Metrics
# -------------------------------------------------------------
def test_orchestration_metrics_increment_and_duration():
    """Tests 9, 10, 11, 12: Orchestration requests, failures, and duration."""
    record_orchestration_request("completed")
    record_orchestration_request("interrupted")
    record_orchestration_request("error")
    record_orchestration_failure(TimeoutError("request timed out"))
    record_orchestration_duration(1.25, "completed")

    req_pts = get_metric_points("orchestration_requests")
    statuses = [pt.attributes.get("status") for pt in req_pts]
    assert "completed" in statuses
    assert "interrupted" in statuses
    assert "error" in statuses

    fail_pts = get_metric_points("orchestration_failures")
    assert len(fail_pts) >= 1
    assert fail_pts[-1].attributes.get("error_category") == "timeout"

    dur_pts = get_metric_points("orchestration_duration")
    assert len(dur_pts) >= 1
    assert dur_pts[-1].attributes.get("status") == "completed"
    assert dur_pts[-1].sum >= 1.25


# -------------------------------------------------------------
# 4. Agent Metrics
# -------------------------------------------------------------
def test_agent_metrics():
    """Tests 13, 14, 15: Agent node executions, failures, and latency."""
    record_agent_execution("research", "success")
    record_agent_execution("data", "error")
    record_agent_failure("data", RuntimeError("data execution failed"))
    record_agent_duration("research", 0.45)

    exec_pts = get_metric_points("agent_executions")
    assert any(
        pt.attributes.get("agent_name") == "research" and pt.attributes.get("status") == "success"
        for pt in exec_pts
    )
    assert any(
        pt.attributes.get("agent_name") == "data" and pt.attributes.get("status") == "error"
        for pt in exec_pts
    )

    fail_pts = get_metric_points("agent_failures")
    assert any(
        pt.attributes.get("agent_name") == "data"
        and pt.attributes.get("error_category") == "execution_error"
        for pt in fail_pts
    )

    dur_pts = get_metric_points("agent_duration")
    assert any(pt.attributes.get("agent_name") == "research" for pt in dur_pts)


# -------------------------------------------------------------
# 5. LLM Metrics
# -------------------------------------------------------------
def test_llm_metrics_with_token_usage():
    """Tests 16, 17, 18, 20: LLM calls, prompt/completion tokens, latency."""
    record_llm_metrics(
        provider="openai",
        model="gpt-4o-mini",
        latency_seconds=0.35,
        status="success",
        prompt_tokens=150,
        completion_tokens=75,
    )

    call_pts = get_metric_points("llm_calls")
    assert any(
        pt.attributes.get("provider") == "openai"
        and pt.attributes.get("model") == "gpt-4o-mini"
        and pt.attributes.get("status") == "success"
        for pt in call_pts
    )

    tok_pts = get_metric_points("llm_tokens")
    prompt_pts = [
        pt
        for pt in tok_pts
        if pt.attributes.get("token_type") == "prompt" and pt.attributes.get("provider") == "openai"
    ]
    completion_pts = [
        pt
        for pt in tok_pts
        if pt.attributes.get("token_type") == "completion"
        and pt.attributes.get("provider") == "openai"
    ]
    assert len(prompt_pts) >= 1
    assert prompt_pts[-1].value == 150
    assert len(completion_pts) >= 1
    assert completion_pts[-1].value == 75

    lat_pts = get_metric_points("llm_latency")
    assert any(
        pt.attributes.get("provider") == "openai" and pt.attributes.get("model") == "gpt-4o-mini"
        for pt in lat_pts
    )


def test_missing_token_usage_does_not_fabricate_tokens():
    """Test 19: When usage metadata is missing/None, token counter is NOT incremented."""
    initial_tok_pts = len(get_metric_points("llm_tokens"))
    record_llm_metrics(
        provider="mock",
        model="mock-model",
        latency_seconds=0.1,
        status="success",
        prompt_tokens=None,
        completion_tokens=None,
    )
    after_tok_pts = len(get_metric_points("llm_tokens"))
    assert after_tok_pts == initial_tok_pts


# -------------------------------------------------------------
# 6. Tool Metrics
# -------------------------------------------------------------
def test_tool_metrics_native_and_mcp():
    """Tests 25, 26, 27, 28, 29: Tool calls, MCP distinction, failure categories, and latency."""
    record_tool_call("calculator", "native", "success")
    record_tool_latency("calculator", "native", 0.015)
    record_tool_call("mcp.local.fetch", "mcp", "error")
    record_tool_failure("mcp.local.fetch", TimeoutError("MCP timeout"))
    record_tool_latency("mcp.local.fetch", "mcp", 5.0)

    call_pts = get_metric_points("tool_calls")
    assert any(
        pt.attributes.get("tool_name") == "calculator"
        and pt.attributes.get("tool_type") == "native"
        and pt.attributes.get("status") == "success"
        for pt in call_pts
    )
    assert any(
        pt.attributes.get("tool_name") == "mcp.local.fetch"
        and pt.attributes.get("tool_type") == "mcp"
        and pt.attributes.get("status") == "error"
        for pt in call_pts
    )

    fail_pts = get_metric_points("tool_failures")
    assert any(
        pt.attributes.get("tool_name") == "mcp.local.fetch"
        and pt.attributes.get("error_category") == "timeout"
        for pt in fail_pts
    )

    lat_pts = get_metric_points("tool_latency")
    assert any(
        pt.attributes.get("tool_name") == "calculator"
        and pt.attributes.get("tool_type") == "native"
        for pt in lat_pts
    )


# -------------------------------------------------------------
# 7. RAG Metrics
# -------------------------------------------------------------
def test_rag_metrics():
    """Tests 30, 31, 32, 33: RAG retrieval, latency, chunks retrieved."""
    record_rag_retrieval("hybrid", "success")
    record_rag_latency("hybrid", 0.08)
    record_rag_chunks_retrieved("hybrid", 5)

    ret_pts = get_metric_points("rag_retrievals")
    assert any(
        pt.attributes.get("strategy") == "hybrid" and pt.attributes.get("status") == "success"
        for pt in ret_pts
    )

    chunks_pts = get_metric_points("rag_chunks_retrieved")
    assert any(pt.attributes.get("strategy") == "hybrid" and pt.value == 5 for pt in chunks_pts)

    lat_pts = get_metric_points("rag_retrieval_latency")
    assert any(pt.attributes.get("strategy") == "hybrid" for pt in lat_pts)


# -------------------------------------------------------------
# 8. Memory Metrics
# -------------------------------------------------------------
def test_memory_metrics():
    """Tests 34, 35, 36: Memory searches, hit/miss flags without scope_id."""
    record_memory_search("success")
    record_memory_hits(True)
    record_memory_hits(False)

    search_pts = get_metric_points("memory_searches")
    assert any(pt.attributes.get("status") == "success" for pt in search_pts)

    hits_pts = get_metric_points("memory_hits")
    hit_vals = [pt.attributes.get("has_hits") for pt in hits_pts]
    assert "true" in hit_vals
    assert "false" in hit_vals


# -------------------------------------------------------------
# 9. HITL Metrics
# -------------------------------------------------------------
def test_hitl_metrics():
    """Tests 37, 38, 39: Escalation, decision, approval latency."""
    record_hitl_escalation("approve_action", "sensitive_action")
    record_hitl_decision("approve")
    record_hitl_decision("reject")
    record_hitl_approval_latency("approve", 12.5)

    esc_pts = get_metric_points("hitl_escalations")
    assert any(
        pt.attributes.get("approval_level") == "approve_action"
        and pt.attributes.get("reason_category") == "sensitive_action"
        for pt in esc_pts
    )

    dec_pts = get_metric_points("hitl_decisions")
    decisions = [pt.attributes.get("decision") for pt in dec_pts]
    assert "approve" in decisions
    assert "reject" in decisions

    lat_pts = get_metric_points("hitl_approval_latency")
    assert any(pt.attributes.get("decision") == "approve" for pt in lat_pts)


# -------------------------------------------------------------
# 10. End-to-End Orchestration & Trace/Metric Coexistence
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestration_execution_emits_expected_metrics():
    """Tests 40, 41, 42, 43: Real orchestration execution emits metrics and coexist with traces."""
    from src.app.observability import get_telemetry_manager
    from src.app.services.orchestration_service import run_orchestrated_task

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}'),
            LLMResponse(content="Task completed deterministically."),
        ]
    )

    settings = Settings(
        telemetry_enabled=True,
        telemetry_exporter="memory",
        checkpoint_backend="memory",
        memory_enabled=False,
    )

    response = await run_orchestrated_task(
        task="Test deterministic workflow",
        settings=settings,
        provider=mock_llm,
    )

    assert response.status == "completed"

    # Verify metrics emitted
    req_pts = get_metric_points("orchestration_requests")
    assert any(pt.attributes.get("status") == "completed" for pt in req_pts)

    dur_pts = get_metric_points("orchestration_duration")
    assert any(pt.attributes.get("status") == "completed" for pt in dur_pts)

    agent_pts = get_metric_points("agent_executions")
    assert any(pt.attributes.get("agent_name") == "final" for pt in agent_pts)

    llm_pts = get_metric_points("llm_calls")
    assert any(pt.attributes.get("provider") == "mock" for pt in llm_pts)

    # Verify traces still coexist
    spans = get_telemetry_manager().get_in_memory_spans()
    span_names = [s.name for s in spans]
    assert "orchestration.run" in span_names
    assert "supervisor.decide_route" in span_names
    assert "final.synthesis" in span_names
