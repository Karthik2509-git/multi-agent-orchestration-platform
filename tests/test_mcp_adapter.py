"""Tests for MCPToolAdapter."""

import pytest

from src.app.mcp.adapters import MCPToolAdapter
from src.app.mcp.client import MCPClient
from src.app.mcp.models import MCPToolDefinition
from src.app.mcp.servers.local_tools import create_local_mcp_server
from src.app.tools.base import BaseTool


@pytest.mark.asyncio
async def test_mcp_tool_adapter_implements_basetool():
    """Verify MCPToolAdapter satisfies BaseTool contract and executes through MCP."""
    server = create_local_mcp_server(name="local")
    client = MCPClient(server_target=server, server_name="local")

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

    adapter = MCPToolAdapter(definition=defn, client=client)

    assert isinstance(adapter, BaseTool)
    assert adapter.name == "mcp.local.calculator"
    assert adapter.description == "Safe arithmetic evaluation"

    # Verify OpenAI function calling schema generation
    schema = adapter.to_openai_schema()
    assert schema["name"] == "mcp.local.calculator"
    assert schema["description"] == "Safe arithmetic evaluation"
    assert "expression" in schema["parameters"]["properties"]

    # Execute tool
    result = await adapter.execute(expression="25 * 4")
    assert result.success is True
    assert result.data == {"result": "100"} or result.data == "100"

    await client.close()
