"""Comprehensive tests for Phase 7 Milestone 3 ToolExecutionContext.

Verifies:
- Global run-scoped tool budget
- 11th attempted call blocked before execution
- Consecutive failure isolation and streak resets
- 3 consecutive failure disablement
- Run-scoped lifecycle
- Multi-agent shared budget
- MCP tool integration with shared budget
- Async concurrency safety
- Controlled error categories and observability integration
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.app.agents.code_agent import CodeAgent
from src.app.agents.data_agent import DataAgent
from src.app.agents.research_agent import ResearchAgent
from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.observability.telemetry import (
    get_telemetry_manager,
    init_telemetry,
    reset_metrics,
    shutdown_telemetry,
)
from src.app.tools.base import BaseTool, ToolResult
from src.app.tools.calculator import CalculatorTool
from src.app.tools.execution_context import ToolExecutionContext
from src.app.tools.http_tool import SafeHTTPGetTool
from src.app.tools.registry import ToolRegistry


class CountingTool(BaseTool):
    """Deterministic dummy tool with execution counter."""

    name: str = "counter_tool"
    description: str = "Increments execution counter and returns success."
    parameters_schema: dict = {
        "type": "object",
        "properties": {"step": {"type": "integer"}},
    }

    def __init__(self, should_fail: bool = False):
        self.execution_count = 0
        self.should_fail = should_fail

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.execution_count += 1
        if self.should_fail:
            return ToolResult(success=False, error="Simulated tool failure")
        return ToolResult(success=True, data={"count": self.execution_count})


class FlakyTool(BaseTool):
    """Tool that fails or succeeds according to a sequence of booleans."""

    name: str = "flaky_tool"
    description: str = "Executes sequence of outcomes."
    parameters_schema: dict = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, outcomes: list[bool]):
        self.outcomes = list(outcomes)
        self.execution_count = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.execution_count += 1
        if self.outcomes:
            success = self.outcomes.pop(0)
        else:
            success = True

        if success:
            return ToolResult(success=True, data="ok")
        return ToolResult(success=False, error="Simulated failure")


# --------------------------------------------------------------------------
# 1. Budget Tests
# --------------------------------------------------------------------------
def test_tool_execution_context_defaults_and_validation():
    """Test 1: Default budget is 10, positive validation enforced."""
    ctx = ToolExecutionContext()
    assert ctx.max_tool_calls == 10
    assert ctx.consecutive_failure_threshold == 3
    assert ctx.tool_calls_attempted == 0
    assert ctx.tool_calls_succeeded == 0
    assert ctx.tool_calls_failed == 0
    assert ctx.disabled_tools == set()

    with pytest.raises(ValueError, match="positive"):
        ToolExecutionContext(max_tool_calls=0)

    with pytest.raises(ValueError, match="positive"):
        ToolExecutionContext(max_tool_calls=-5)

    with pytest.raises(ValueError, match="positive"):
        ToolExecutionContext(consecutive_failure_threshold=0)


def test_custom_budget_configuration():
    """Test 2: Custom budget configuration."""
    ctx = ToolExecutionContext(max_tool_calls=5, consecutive_failure_threshold=2)
    assert ctx.max_tool_calls == 5
    assert ctx.consecutive_failure_threshold == 2


@pytest.mark.asyncio
async def test_eleventh_attempt_blocked_before_execution():
    """Test 4, 5, 29: Calls 1-10 allowed, 11th call blocked before execution."""
    registry = ToolRegistry()
    counter_tool = CountingTool()
    registry.register(counter_tool)

    ctx = ToolExecutionContext(max_tool_calls=10)

    # Execute 10 allowed calls
    for i in range(1, 11):
        res = await registry.execute("counter_tool", {"step": i}, context=ctx)
        assert res.success is True
        assert res.error is None
        assert counter_tool.execution_count == i

    assert counter_tool.execution_count == 10
    assert ctx.tool_calls_attempted == 10
    assert ctx.tool_calls_succeeded == 10

    # 11th call attempt must be BLOCKED before executing underlying tool
    res_11 = await registry.execute("counter_tool", {"step": 11}, context=ctx)
    assert res_11.success is False
    assert res_11.error_category == "budget_exceeded"
    assert "budget exceeded" in res_11.error.lower()

    # Crucial assertion: Underlying tool's execute() was NEVER invoked for call 11
    assert counter_tool.execution_count == 10
    assert ctx.tool_calls_attempted == 11
    assert ctx.tool_calls_failed == 1


@pytest.mark.asyncio
async def test_failed_calls_still_consume_budget():
    """Test 6: Failed calls consume attempted budget count."""
    registry = ToolRegistry()
    failing_tool = CountingTool(should_fail=True)
    registry.register(failing_tool)

    ctx = ToolExecutionContext(max_tool_calls=2)

    res1 = await registry.execute("counter_tool", {}, context=ctx)
    assert res1.success is False
    assert ctx.tool_calls_attempted == 1

    res2 = await registry.execute("counter_tool", {}, context=ctx)
    assert res2.success is False
    assert ctx.tool_calls_attempted == 2

    # 3rd attempt is blocked by budget
    res3 = await registry.execute("counter_tool", {}, context=ctx)
    assert res3.success is False
    assert res3.error_category == "budget_exceeded"
    assert ctx.tool_calls_attempted == 3
    assert failing_tool.execution_count == 2


@pytest.mark.asyncio
async def test_budget_is_shared_across_different_tools():
    """Test 8: Budget is shared across different tools within the same context."""
    registry = ToolRegistry()
    calc = CalculatorTool()
    counter = CountingTool()
    registry.register(calc)
    registry.register(counter)

    ctx = ToolExecutionContext(max_tool_calls=3)

    res1 = await registry.execute("calculator", {"expression": "2 + 2"}, context=ctx)
    res2 = await registry.execute("counter_tool", {"step": 1}, context=ctx)
    res3 = await registry.execute("calculator", {"expression": "10 / 2"}, context=ctx)

    assert res1.success is True
    assert res2.success is True
    assert res3.success is True
    assert ctx.tool_calls_attempted == 3

    # 4th call to either tool is blocked
    res4 = await registry.execute("counter_tool", {"step": 2}, context=ctx)
    assert res4.success is False
    assert res4.error_category == "budget_exceeded"
    assert counter.execution_count == 1


# --------------------------------------------------------------------------
# 2. Failure Isolation & Consecutive Disablement Tests
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_consecutive_failure_disablement_and_fourth_attempt_blocked():
    """Test 11, 12, 13, 14, 15: 3 consecutive failures disable tool; 4th attempt blocked."""
    registry = ToolRegistry()
    failing_tool = CountingTool(should_fail=True)
    registry.register(failing_tool)

    ctx = ToolExecutionContext(max_tool_calls=10, consecutive_failure_threshold=3)

    # Failure 1
    res1 = await registry.execute("counter_tool", {}, context=ctx)
    assert res1.success is False
    assert ctx.get_consecutive_failures("counter_tool") == 1
    assert ctx.is_disabled("counter_tool") is False

    # Failure 2
    res2 = await registry.execute("counter_tool", {}, context=ctx)
    assert res2.success is False
    assert ctx.get_consecutive_failures("counter_tool") == 2
    assert ctx.is_disabled("counter_tool") is False

    # Failure 3 -> triggers disablement
    res3 = await registry.execute("counter_tool", {}, context=ctx)
    assert res3.success is False
    assert ctx.get_consecutive_failures("counter_tool") == 3
    assert ctx.is_disabled("counter_tool") is True
    assert failing_tool.execution_count == 3

    # Attempt 4 -> BLOCKED before execution
    res4 = await registry.execute("counter_tool", {}, context=ctx)
    assert res4.success is False
    assert res4.error_category == "tool_disabled"
    assert "disabled" in res4.error.lower()

    # Crucial assertion: tool was NOT executed on 4th attempt
    assert failing_tool.execution_count == 3


@pytest.mark.asyncio
async def test_successful_execution_resets_consecutive_failure_streak():
    """Test 16, 30: Sequence (FAIL, FAIL, SUCCESS, FAIL, FAIL, FAIL) resets streak."""
    registry = ToolRegistry()
    flaky = FlakyTool(outcomes=[False, False, True, False, False, False])
    registry.register(flaky)

    ctx = ToolExecutionContext(max_tool_calls=10, consecutive_failure_threshold=3)

    # 1. Fail
    await registry.execute("flaky_tool", {}, context=ctx)
    assert ctx.get_consecutive_failures("flaky_tool") == 1

    # 2. Fail
    await registry.execute("flaky_tool", {}, context=ctx)
    assert ctx.get_consecutive_failures("flaky_tool") == 2
    assert ctx.is_disabled("flaky_tool") is False

    # 3. Success -> STREAK RESETS TO 0!
    res_success = await registry.execute("flaky_tool", {}, context=ctx)
    assert res_success.success is True
    assert ctx.get_consecutive_failures("flaky_tool") == 0
    assert ctx.is_disabled("flaky_tool") is False

    # 4. Fail (streak 1)
    await registry.execute("flaky_tool", {}, context=ctx)
    assert ctx.get_consecutive_failures("flaky_tool") == 1

    # 5. Fail (streak 2)
    await registry.execute("flaky_tool", {}, context=ctx)
    assert ctx.get_consecutive_failures("flaky_tool") == 2
    assert ctx.is_disabled("flaky_tool") is False

    # 6. Fail (streak 3 -> now disabled!)
    await registry.execute("flaky_tool", {}, context=ctx)
    assert ctx.get_consecutive_failures("flaky_tool") == 3
    assert ctx.is_disabled("flaky_tool") is True

    # Next attempt is blocked as disabled
    res_blocked = await registry.execute("flaky_tool", {}, context=ctx)
    assert res_blocked.error_category == "tool_disabled"
    assert flaky.execution_count == 6


@pytest.mark.asyncio
async def test_independent_tool_failure_streaks():
    """Test 17: Different tools maintain completely independent failure streaks."""
    registry = ToolRegistry()
    t1 = CountingTool(should_fail=True)
    t1.name = "tool_1"
    t2 = CountingTool(should_fail=False)
    t2.name = "tool_2"
    registry.register(t1)
    registry.register(t2)

    ctx = ToolExecutionContext(max_tool_calls=10, consecutive_failure_threshold=3)

    # tool_1 fails twice
    await registry.execute("tool_1", {}, context=ctx)
    await registry.execute("tool_1", {}, context=ctx)

    # tool_2 succeeds
    await registry.execute("tool_2", {}, context=ctx)

    assert ctx.get_consecutive_failures("tool_1") == 2
    assert ctx.get_consecutive_failures("tool_2") == 0
    assert ctx.is_disabled("tool_1") is False
    assert ctx.is_disabled("tool_2") is False


@pytest.mark.asyncio
async def test_disabled_state_is_run_scoped():
    """Test 18: Disabled state applies only to that context; new run starts enabled."""
    registry = ToolRegistry()
    failing_tool = CountingTool(should_fail=True)
    registry.register(failing_tool)

    ctx1 = ToolExecutionContext(max_tool_calls=5, consecutive_failure_threshold=2)
    await registry.execute("counter_tool", {}, context=ctx1)
    await registry.execute("counter_tool", {}, context=ctx1)
    assert ctx1.is_disabled("counter_tool") is True

    # New orchestration run with new context
    ctx2 = ToolExecutionContext(max_tool_calls=5, consecutive_failure_threshold=2)
    assert ctx2.is_disabled("counter_tool") is False


# --------------------------------------------------------------------------
# 3. Multi-Agent Shared Budget Test
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shared_multi_agent_budget():
    """Test 9, 31: ResearchAgent (4) + DataAgent (3) + CodeAgent (3) = 10 calls.

    11th call from any agent is blocked.
    """
    registry = ToolRegistry(
        mock_tool_results={
            "http_get": {
                "success": True,
                "data": {"content": "Deterministic offline fixture for shared-budget testing."},
            }
        }
    )
    counter_tool = CountingTool()
    registry.register(counter_tool)

    calc = CalculatorTool()
    http_tool = SafeHTTPGetTool(allowed_domains=["example.com"])
    registry.register(calc)
    registry.register(http_tool)

    ctx = ToolExecutionContext(max_tool_calls=10)
    state: dict[str, Any] = {"tool_execution_context": ctx}

    # 1. ResearchAgent executes 4 tool calls
    research_agent = ResearchAgent(
        provider=MockLLMProvider(responses=[LLMResponse(content="Research synthesis")]),
        http_tool=http_tool,
        registry=registry,
    )
    for _ in range(4):
        await research_agent.run("Inspect https://example.com/test", state)
    assert ctx.tool_calls_attempted == 4

    # 2. DataAgent executes 3 tool calls
    data_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Calculation done",
                tool_calls=[
                    ToolCall(id="c1", name="calculator", arguments={"expression": "1 + 1"})
                ],
            ),
            LLMResponse(
                content="Calculation done",
                tool_calls=[
                    ToolCall(id="c2", name="calculator", arguments={"expression": "2 + 2"})
                ],
            ),
            LLMResponse(
                content="Calculation done",
                tool_calls=[
                    ToolCall(id="c3", name="calculator", arguments={"expression": "3 + 3"})
                ],
            ),
        ]
    )
    data_agent = DataAgent(
        provider=data_provider,
        calculator=calc,
        registry=registry,
    )
    for _ in range(3):
        await data_agent.run("Calculate", state)
    assert ctx.tool_calls_attempted == 7

    # 3. CodeAgent executes 3 tool calls
    code_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Code done",
                tool_calls=[
                    ToolCall(id="k1", name="calculator", arguments={"expression": "4 + 4"})
                ],
            ),
            LLMResponse(
                content="Code done",
                tool_calls=[
                    ToolCall(id="k2", name="calculator", arguments={"expression": "5 + 5"})
                ],
            ),
            LLMResponse(
                content="Code done",
                tool_calls=[
                    ToolCall(id="k3", name="calculator", arguments={"expression": "6 + 6"})
                ],
            ),
        ]
    )
    code_agent = CodeAgent(
        provider=code_provider,
        registry=registry,
    )
    for _ in range(3):
        await code_agent.run("Code task", state)
    assert ctx.tool_calls_attempted == 10

    # 4. Attempt 11th tool call via DataAgent
    blocked_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Extra calc",
                tool_calls=[
                    ToolCall(id="c4", name="calculator", arguments={"expression": "7 + 7"})
                ],
            )
        ]
    )
    extra_agent = DataAgent(provider=blocked_provider, calculator=calc, registry=registry)
    result = await extra_agent.run("One more calculation", state)
    assert ctx.tool_calls_attempted == 11
    assert "budget exceeded" in result.result.lower()


# --------------------------------------------------------------------------
# 4. MCP Tools with Shared Budget
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mcp_tool_obeys_global_budget_and_failure_streak():
    """Test 19, 20, 21: MCP tool obeys global budget and failure streak."""
    from src.app.mcp.adapters import MCPToolAdapter
    from src.app.mcp.client import MCPClient
    from src.app.mcp.models import MCPToolDefinition

    mock_client = AsyncMock(spec=MCPClient)
    mock_client.call_tool.return_value = ToolResult(success=True, data={"result": "mcp_ok"})

    mcp_def = MCPToolDefinition(
        name="mcp.local.fetch",
        original_name="fetch",
        server_name="local",
        description="MCP fetch tool",
        input_schema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(definition=mcp_def, client=mock_client)

    registry = ToolRegistry()
    registry.register(adapter)

    ctx = ToolExecutionContext(max_tool_calls=2)

    # Call 1: permitted
    res1 = await registry.execute("mcp.local.fetch", {}, context=ctx)
    assert res1.success is True
    assert ctx.tool_calls_attempted == 1

    # Call 2: permitted
    res2 = await registry.execute("mcp.local.fetch", {}, context=ctx)
    assert res2.success is True
    assert ctx.tool_calls_attempted == 2

    # Call 3: blocked by global budget
    res3 = await registry.execute("mcp.local.fetch", {}, context=ctx)
    assert res3.success is False
    assert res3.error_category == "budget_exceeded"
    assert mock_client.call_tool.call_count == 2


# --------------------------------------------------------------------------
# 5. Concurrency Safety
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_tool_execution_respects_budget():
    """Test 30: Concurrent tool executions cannot exceed max_tool_calls."""
    registry = ToolRegistry()
    counter_tool = CountingTool()
    registry.register(counter_tool)

    ctx = ToolExecutionContext(max_tool_calls=5)

    # Fire 15 concurrent calls
    tasks = [registry.execute("counter_tool", {"step": i}, context=ctx) for i in range(15)]
    results = await asyncio.gather(*tasks)

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    # Exactly 5 should succeed, 10 should be blocked by budget
    assert len(successes) == 5
    assert len(failures) == 10
    assert counter_tool.execution_count == 5
    assert ctx.tool_calls_succeeded == 5
    assert ctx.tool_calls_attempted == 15
    for f in failures:
        assert f.error_category == "budget_exceeded"


# --------------------------------------------------------------------------
# 6. Observability Integration
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_observability_metric_recording_on_budget_and_disablement():
    """Test 32, 33, 35, 36: Metric points emitted with low-cardinality categories."""
    init_telemetry(Settings(telemetry_enabled=True, telemetry_exporter="memory"))
    reset_metrics()
    mgr = get_telemetry_manager()

    registry = ToolRegistry()
    failing = CountingTool(should_fail=True)
    registry.register(failing)

    ctx = ToolExecutionContext(max_tool_calls=1, consecutive_failure_threshold=1)

    # Call 1 fails -> triggers disablement
    await registry.execute("counter_tool", {}, context=ctx)

    # Call 2 -> blocked as tool_disabled
    await registry.execute("counter_tool", {}, context=ctx)

    pts = mgr.get_metric_data_points("tool_failures")
    categories = [pt.attributes.get("error_category") for pt in pts]

    assert "tool_disabled" in categories

    # Cardinality safety: no prohibited attributes
    prohibited = {"thread_id", "task_id", "trace_id", "span_id", "memory_id"}
    for pt in pts:
        for p in prohibited:
            assert p not in pt.attributes

    shutdown_telemetry()
