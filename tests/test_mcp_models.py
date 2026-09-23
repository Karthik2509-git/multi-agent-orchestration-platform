"""Tests for normalized MCP data models."""

from src.app.mcp.models import MCPServerStatus, MCPToolDefinition


def test_mcp_tool_definition_model():
    """Verify MCPToolDefinition fields and serialization."""
    defn = MCPToolDefinition(
        name="mcp.local.calculator",
        server_name="local",
        original_name="calculator",
        description="Safe arithmetic evaluation",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )

    assert defn.name == "mcp.local.calculator"
    assert defn.server_name == "local"
    assert defn.original_name == "calculator"
    assert defn.source == "mcp"
    assert "expression" in defn.input_schema["properties"]


def test_mcp_server_status_model():
    """Verify MCPServerStatus data contract."""
    status = MCPServerStatus(
        server_name="local",
        status="connected",
        tools_count=2,
    )

    assert status.server_name == "local"
    assert status.status == "connected"
    assert status.tools_count == 2
    assert status.error is None
