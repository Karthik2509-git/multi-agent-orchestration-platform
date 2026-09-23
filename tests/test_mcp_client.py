"""Tests for MCPClient wrapper using official MCP SDK v2 Client."""

import pytest

from src.app.mcp.client import MCPClient
from src.app.mcp.servers.local_tools import create_local_mcp_server


@pytest.mark.asyncio
async def test_mcp_client_connect_and_list_tools():
    """Verify MCPClient connects via MCP v2 Client and discovers tools over in-process transport."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

    assert not client.is_connected
    await client.connect()
    assert client.is_connected

    tools = await client.list_tools()
    assert len(tools) >= 2

    tool_names = [t.name for t in tools]
    assert "mcp.local.calculator" in tool_names
    assert "mcp.local.text_stats" in tool_names

    calc_def = next(t for t in tools if t.name == "mcp.local.calculator")
    assert calc_def.original_name == "calculator"
    assert calc_def.server_name == "local"
    assert calc_def.source == "mcp"
    assert "expression" in calc_def.input_schema.get("properties", {})

    await client.close()
    assert not client.is_connected


@pytest.mark.asyncio
async def test_mcp_client_call_tool_success():
    """Verify MCPClient successfully invokes tools and normalizes output to ToolResult."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

    res = await client.call_tool(original_name="calculator", arguments={"expression": "12 * 8"})
    assert res.success is True
    assert res.data == {"result": "96"} or res.data == "96"
    assert res.error is None

    stats_res = await client.call_tool(
        original_name="text_stats",
        arguments={"text": "Hello world from MCP platform"},
    )
    assert stats_res.success is True
    assert "Words: 5" in str(stats_res.data)

    await client.close()


@pytest.mark.asyncio
async def test_mcp_client_call_tool_error_handling():
    """Verify MCP tool errors are caught and returned safely as ToolResult(success=False)."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

    # Call with non-existent tool name
    res = await client.call_tool(original_name="nonexistent_tool", arguments={})
    assert res.success is False
    assert res.error is not None

    await client.close()
