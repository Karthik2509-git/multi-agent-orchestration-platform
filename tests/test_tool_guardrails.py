"""Comprehensive tests for Phase 7 Milestone 3 Structural Guardrails.

Verifies:
- Structural input validation (schema, types, required parameters, bounds, enums)
- Input size limits and string length bounds
- URL structural safety (scheme and host validation)
- Output size limits and serialization safety
- Underlying tool is NOT executed for invalid inputs
- Absence of prompt-injection keyword heuristics
- Defense-in-depth preservation of SafeHTTPGetTool
- Observability integration with controlled error categories
"""

from typing import Any

import pytest

from src.app.tools.base import BaseTool, ToolResult
from src.app.tools.calculator import CalculatorTool
from src.app.tools.guardrails import ToolGuardrails
from src.app.tools.http_tool import SafeHTTPGetTool
from src.app.tools.registry import ToolRegistry


class DummySchemaTool(BaseTool):
    """Tool with detailed parameter schema for testing structural guardrails."""

    name: str = "schema_tool"
    description: str = "Tool with explicit type, bounds, and enum schemas."
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 50},
            "ratio": {"type": "number"},
            "mode": {"type": "string", "enum": ["fast", "deep", "hybrid"]},
            "target_url": {"type": "string"},
        },
        "required": ["query"],
    }

    def __init__(self):
        self.invoked = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.invoked = True
        return ToolResult(success=True, data={"processed": kwargs})


class GiantOutputTool(BaseTool):
    """Tool that returns output of a configurable size or unserializable data."""

    name: str = "giant_tool"
    description: str = "Tool producing large or custom output."
    parameters_schema: dict = {"type": "object", "properties": {}}

    def __init__(self, output_data: Any):
        self.output_data = output_data

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=self.output_data)


# --------------------------------------------------------------------------
# 1. Structural Input Guardrails
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_required_argument_rejected_before_execution():
    """Test 22, 23: Missing required parameter rejected; underlying tool not invoked."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    # Missing "query"
    res = await registry.execute("schema_tool", {"count": 10})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "Missing required parameter 'query'" in res.error
    assert tool.invoked is False


@pytest.mark.asyncio
async def test_invalid_type_rejected_before_execution():
    """Test 22, 23: Invalid argument type rejected before execution."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    # "query" should be string, got int
    res = await registry.execute("schema_tool", {"query": 12345})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "Invalid type for parameter 'query'" in res.error
    assert tool.invoked is False

    # "count" should be integer, got string
    res2 = await registry.execute("schema_tool", {"query": "test", "count": "not_an_int"})
    assert res2.success is False
    assert res2.error_category == "validation_error"
    assert "Invalid type for parameter 'count'" in res2.error
    assert tool.invoked is False


@pytest.mark.asyncio
async def test_enum_allowlist_validation():
    """Test 22: Value outside enum allowlist is rejected before execution."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    # "mode" must be one of ["fast", "deep", "hybrid"]
    res = await registry.execute("schema_tool", {"query": "test", "mode": "invalid_mode"})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "must be one of" in res.error
    assert tool.invoked is False


@pytest.mark.asyncio
async def test_numeric_bounds_validation():
    """Test 22: Numeric minimum/maximum bounds enforced."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    # count < minimum (1)
    res_min = await registry.execute("schema_tool", {"query": "test", "count": 0})
    assert res_min.success is False
    assert res_min.error_category == "validation_error"
    assert "cannot be less than 1" in res_min.error
    assert tool.invoked is False

    # count > maximum (50)
    res_max = await registry.execute("schema_tool", {"query": "test", "count": 51})
    assert res_max.success is False
    assert res_max.error_category == "validation_error"
    assert "cannot be greater than 50" in res_max.error
    assert tool.invoked is False


@pytest.mark.asyncio
async def test_valid_input_executes_normally():
    """Test 24: Valid structured input executes successfully."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    res = await registry.execute(
        "schema_tool",
        {
            "query": "valid search",
            "count": 5,
            "mode": "fast",
        },
    )
    assert res.success is True
    assert tool.invoked is True
    assert res.data["processed"]["query"] == "valid search"


# --------------------------------------------------------------------------
# 2. Input Size Limits
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_oversized_input_payload_rejected():
    """Test 25: Oversized input payload exceeding max_input_size_bytes is rejected."""
    guardrails = ToolGuardrails(max_input_size_bytes=500)
    registry = ToolRegistry(guardrails=guardrails)
    tool = DummySchemaTool()
    registry.register(tool)

    huge_query = "x" * 600
    res = await registry.execute("schema_tool", {"query": huge_query})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "exceeds maximum allowed size" in res.error
    assert tool.invoked is False


@pytest.mark.asyncio
async def test_oversized_string_argument_rejected():
    """Test 25: Individual string field exceeding max_string_length is rejected."""
    guardrails = ToolGuardrails(max_input_size_bytes=100_000, max_string_length=100)
    registry = ToolRegistry(guardrails=guardrails)
    tool = DummySchemaTool()
    registry.register(tool)

    long_str = "a" * 150
    res = await registry.execute("schema_tool", {"query": long_str})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "exceeds maximum length" in res.error
    assert tool.invoked is False


# --------------------------------------------------------------------------
# 3. URL Structural Safety
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_url_structural_safety_accepts_valid_urls():
    """Test 22: Valid HTTP/HTTPS URLs pass structural guardrails."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    res = await registry.execute(
        "schema_tool",
        {"query": "check", "target_url": "https://api.github.com/zen"},
    )
    assert res.success is True
    assert tool.invoked is True


@pytest.mark.asyncio
async def test_url_structural_safety_rejects_disallowed_schemes():
    """Test 22: File, javascript, and data schemes are structurally rejected."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    for bad_url in [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html;base64,PHNjcmlwdD4=",
        "gopher://evil.com",
    ]:
        tool.invoked = False
        res = await registry.execute(
            "schema_tool",
            {"query": "check", "target_url": bad_url},
        )
        assert res.success is False
        assert res.error_category == "validation_error"
        assert "Only HTTP/HTTPS are supported" in res.error
        assert tool.invoked is False


@pytest.mark.asyncio
async def test_url_structural_safety_rejects_missing_host_and_control_chars():
    """Test 22: Missing host or control characters rejected."""
    registry = ToolRegistry()
    tool = DummySchemaTool()
    registry.register(tool)

    # Missing host
    res = await registry.execute("schema_tool", {"query": "check", "target_url": "http://"})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "must contain a valid host" in res.error

    # Control character injection
    res2 = await registry.execute(
        "schema_tool", {"query": "check", "target_url": "https://example.com/\r\nHeader: inject"}
    )
    assert res2.success is False
    assert res2.error_category == "validation_error"
    assert "illegal control characters" in res2.error


# --------------------------------------------------------------------------
# 4. Structural Output Guardrails
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_oversized_output_rejected_safely():
    """Test 26: Output exceeding max_output_size_bytes returns controlled failure."""
    guardrails = ToolGuardrails(max_output_size_bytes=500)
    registry = ToolRegistry(guardrails=guardrails)
    giant_tool = GiantOutputTool(output_data={"blob": "A" * 1000})
    registry.register(giant_tool)

    res = await registry.execute("giant_tool", {})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "output exceeded maximum allowed size" in res.error


@pytest.mark.asyncio
async def test_unserializable_output_rejected_safely():
    """Test 27: Unserializable output produces controlled failure without crashing."""
    guardrails = ToolGuardrails()
    registry = ToolRegistry(guardrails=guardrails)

    # Non-serializable object (lambda)
    unserializable_data = {"fn": lambda x: x}
    tool = GiantOutputTool(output_data=unserializable_data)
    registry.register(tool)

    res = await registry.execute("giant_tool", {})
    assert res.success is False
    assert res.error_category == "validation_error"
    assert "non-serializable output" in res.error


# --------------------------------------------------------------------------
# 5. Non-Goal: No Prompt Injection Keyword Blacklist
# --------------------------------------------------------------------------
def test_no_prompt_injection_keyword_heuristics():
    """Test 29: Confirm NO prompt-injection keyword blacklist heuristic exists.

    Valid user inputs containing words like 'system prompt' or 'ignore instructions'
    must not be rejected by the structural guardrail layer.
    """
    guardrails = ToolGuardrails()
    calc = CalculatorTool()

    # Pass a valid expression or note containing phrases that a brittle blacklist would reject
    args = {"expression": "2 + 2"}
    err = guardrails.validate_input(calc, args)
    assert err is None

    # Inspect source code of guardrails to verify absence of blacklisted strings
    import inspect

    source = inspect.getsource(guardrails.__class__)
    forbidden_heuristics = [
        "ignore previous",
        "system prompt",
        "jailbreak",
        "developer message",
        "reveal instructions",
    ]
    for heuristic in forbidden_heuristics:
        assert heuristic not in source.lower()


# --------------------------------------------------------------------------
# 6. Defense-In-Depth: SafeHTTPGetTool SSRF Protections Intact
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_safe_http_tool_ssrf_protections_intact():
    """Test 28: Existing SafeHTTPGetTool SSRF protections remain fully intact."""
    tool = SafeHTTPGetTool(allowed_domains=["httpbin.org"])

    # Disallowed domain rejected
    res_domain = await tool.execute(url="https://evil.internal.corp/admin")
    assert res_domain.success is False
    assert "not in the allowed domains list" in res_domain.error

    # Private IP rejected
    res_ip = await tool.execute(url="http://192.168.1.1/secret")
    assert res_ip.success is False
