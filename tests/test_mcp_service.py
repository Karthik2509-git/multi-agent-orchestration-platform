"""Tests for MCPService lifecycle and status reporting."""

import pytest

from src.app.core.config import Settings
from src.app.services.mcp_service import MCPService
from src.app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_service_disabled_by_default():
    """Verify MCPService remains inactive when mcp_enabled is False."""
    settings = Settings(mcp_enabled=False)
    service = MCPService(settings=settings)

    assert not service.is_enabled
    await service.initialize()

    health = service.get_health()
    assert health.enabled is False
    assert health.servers == {}

    tools = service.get_tools_list()
    assert tools.servers == []

    tool_registry = ToolRegistry()
    service.register_tools_into(tool_registry)
    assert len(tool_registry.list_tools()) == 0

    await service.shutdown()


@pytest.mark.asyncio
async def test_mcp_service_enabled_lifecycle():
    """Verify MCPService initialization, discovery, and tool registration when enabled."""
    settings = Settings(
        mcp_enabled=True,
        mcp_local_server_enabled=True,
        mcp_local_server_name="local",
        mcp_allowed_tools=["mcp.local.calculator", "mcp.local.text_stats"],
    )
    service = MCPService(settings=settings)

    assert service.is_enabled
    await service.initialize()

    health = service.get_health()
    assert health.enabled is True
    assert health.servers.get("local") == "connected"

    tools_response = service.get_tools_list()
    assert len(tools_response.servers) == 1
    local_info = tools_response.servers[0]
    assert local_info.name == "local"
    assert local_info.status == "connected"
    assert len(local_info.tools) == 2

    # Verify tool injection into ToolRegistry
    tool_registry = ToolRegistry()
    service.register_tools_into(tool_registry)
    assert tool_registry.get("mcp.local.calculator") is not None
    assert tool_registry.get("mcp.local.text_stats") is not None

    await service.shutdown()
