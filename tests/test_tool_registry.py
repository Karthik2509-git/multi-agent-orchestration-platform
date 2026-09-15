"""Tests for ToolRegistry lifecycle and execution."""

import pytest

from src.app.tools.base import BaseTool, ToolResult
from src.app.tools.calculator import CalculatorTool
from src.app.tools.registry import ToolRegistry


class DummyFailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool designed to raise an unhandled error."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Simulated crash inside tool")


@pytest.mark.asyncio
async def test_tool_registration_and_lookup() -> None:
    """Test registering tools and looking them up."""
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)
    assert registry.get("calculator") is calculator
    assert len(registry.list_tools()) == 1

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "calculator"
    assert "parameters" in schemas[0]


@pytest.mark.asyncio
async def test_execute_unknown_tool() -> None:
    """Test executing an unknown tool returns a clean error result."""
    registry = ToolRegistry()
    result = await registry.execute("non_existent_tool", {"arg": "value"})

    assert result.success is False
    assert "not registered" in result.error


@pytest.mark.asyncio
async def test_execute_registered_tool() -> None:
    """Test executing a registered tool successfully."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    result = await registry.execute("calculator", {"expression": "12 * 12"})
    assert result.success is True
    assert result.data["result"] == 144


@pytest.mark.asyncio
async def test_tool_exception_isolation() -> None:
    """Test that unexpected exceptions within tools are caught cleanly."""
    registry = ToolRegistry()
    registry.register(DummyFailingTool())

    result = await registry.execute("failing_tool", {})
    assert result.success is False
    assert "Internal error executing tool" in result.error
