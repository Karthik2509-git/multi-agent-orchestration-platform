"""Tests for MCPToolRegistry, allowlisting, and ToolRegistry integration."""

import pytest

from src.app.mcp.client import MCPClient
from src.app.mcp.registry import MCPToolRegistry
from src.app.mcp.servers.local_tools import create_local_mcp_server
from src.app.tools.calculator import CalculatorTool
from src.app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_registry_discovery_and_allowlist():
    """Verify tool discovery filters against allowlist."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

    # Only allow calculator, exclude text_stats
    registry = MCPToolRegistry(allowed_tools=["mcp.local.calculator"])
    registry.register_server(name="local", client=client)

    discovered = await registry.discover_tools()
    assert len(discovered) == 1
    assert discovered[0].name == "mcp.local.calculator"

    adapters = registry.get_tool_adapters()
    assert len(adapters) == 1
    assert adapters[0].name == "mcp.local.calculator"

    await client.close()


@pytest.mark.asyncio
async def test_mcp_registry_injects_into_tool_registry():
    """Verify approved MCP tools are injected into application ToolRegistry."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

    mcp_registry = MCPToolRegistry(allowed_tools=["mcp.local.calculator", "mcp.local.text_stats"])
    mcp_registry.register_server(name="local", client=client)
    await mcp_registry.discover_tools()

    tool_registry = ToolRegistry()
    # Pre-populate native tool
    tool_registry.register(CalculatorTool())

    injected = mcp_registry.register_into_tool_registry(tool_registry)
    assert len(injected) == 2
    assert "mcp.local.calculator" in injected
    assert "mcp.local.text_stats" in injected

    # Verify both native and MCP tools coexist
    assert tool_registry.get("calculator") is not None
    assert tool_registry.get("mcp.local.calculator") is not None
    assert tool_registry.get("mcp.local.text_stats") is not None

    # Execute MCP tool via ToolRegistry
    result = await tool_registry.execute(
        name="mcp.local.calculator",
        arguments={"expression": "100 / 5"},
    )
    assert result.success is True
    assert result.data == {"result": "20"} or result.data == "20"

    await client.close()


@pytest.mark.asyncio
async def test_mcp_registry_collision_rejection():
    """Verify collision with native tool is rejected to prevent silent overwriting."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

    mcp_registry = MCPToolRegistry(allowed_tools=["calculator"])
    mcp_registry.register_server(name="local", client=client)
    await mcp_registry.discover_tools()

    tool_registry = ToolRegistry()
    native_calc = CalculatorTool()
    tool_registry.register(native_calc)

    # Attempt to inject an adapter named exactly 'calculator' (collision)
    from src.app.mcp.adapters import MCPToolAdapter
    from src.app.mcp.models import MCPToolDefinition

    colliding_defn = MCPToolDefinition(
        name="calculator",
        server_name="local",
        original_name="calculator",
        description="Colliding tool",
        input_schema={},
    )
    mcp_registry._adapters["calculator"] = MCPToolAdapter(
        definition=colliding_defn,
        client=client,
    )

    injected = mcp_registry.register_into_tool_registry(tool_registry)
    assert "calculator" not in injected
    # Original native calculator remains untouched
    assert tool_registry.get("calculator") is native_calc

    await client.close()
