"""Integration tests verifying ToolCallingAgent execution of MCP tools via ToolRegistry."""

import pytest

from src.app.agents.tool_calling_agent import ToolCallingAgent
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.services.agent_service import build_default_tool_registry
from src.app.services.mcp_service import MCPService


@pytest.mark.asyncio
async def test_tool_calling_agent_executes_mcp_tool():
    """Verify ToolCallingAgent executes mcp.local.calculator via standard ToolRegistry.

    Architecture under test:
    ToolCallingAgent
           │ (calls standard ToolRegistry.execute)
           ▼
      ToolRegistry
           │ (resolves registered BaseTool)
           ▼
     MCPToolAdapter ('mcp.local.calculator')
           │ (calls MCP client)
           ▼
       MCPClient (official v2 Client)
           │ (in-process MCP protocol transport)
           ▼
      MCPServer ('local')
           │
           ▼
      Tool Result
    """
    from src.app.core.config import Settings

    settings = Settings(
        mcp_enabled=True,
        mcp_local_server_enabled=True,
        mcp_local_server_name="local",
        mcp_allowed_tools=["mcp.local.calculator"],
    )
    mcp_service = MCPService(settings=settings)
    await mcp_service.initialize()

    # Build ToolRegistry with native tools + injected MCP tools
    registry = build_default_tool_registry(settings=settings, mcp_service=mcp_service)

    # Verify mcp.local.calculator is in schemas
    schemas = registry.get_schemas()
    schema_names = [s["name"] for s in schemas]
    assert "mcp.local.calculator" in schema_names
    assert "calculator" in schema_names

    # Configure MockLLMProvider to issue a tool call to 'mcp.local.calculator'
    mock_provider = MockLLMProvider(
        responses=[
            # Step 1: Model requests tool call to mcp.local.calculator
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_mcp_1",
                        name="mcp.local.calculator",
                        arguments={"expression": "(50 * 4) + 12"},
                    )
                ],
            ),
            # Step 2: Model receives tool result and produces final answer
            LLMResponse(
                content="The computed result from the calculation is 212.",
            ),
        ]
    )

    # ToolCallingAgent knows NOTHING about MCP - it only receives the registry and provider
    agent = ToolCallingAgent(
        provider=mock_provider,
        registry=registry,
        max_iterations=5,
    )

    response = await agent.run("Calculate (50 * 4) + 12")

    assert response.status == "completed"
    assert "212" in response.answer
    assert len(response.tool_calls) == 1

    audit_entry = response.tool_calls[0]
    assert audit_entry.tool_name == "mcp.local.calculator"
    assert audit_entry.arguments == {"expression": "(50 * 4) + 12"}
    assert audit_entry.success is True
    assert audit_entry.result == {"result": "212"} or audit_entry.result == "212"

    await mcp_service.shutdown()
