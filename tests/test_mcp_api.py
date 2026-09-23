"""Tests for MCP REST API endpoints (/api/v1/mcp/health and /api/v1/mcp/tools)."""

import pytest
from fastapi.testclient import TestClient

from src.app.core.config import Settings
from src.app.main import create_app
from src.app.services.mcp_service import MCPService, get_mcp_service, reset_mcp_service


def test_mcp_health_endpoint_disabled(client: TestClient) -> None:
    """Test GET /api/v1/mcp/health when MCP is disabled."""
    reset_mcp_service()
    response = client.get("/api/v1/mcp/health")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["servers"] == {}


def test_mcp_tools_endpoint_disabled(client: TestClient) -> None:
    """Test GET /api/v1/mcp/tools when MCP is disabled."""
    reset_mcp_service()
    response = client.get("/api/v1/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["servers"] == []


@pytest.mark.asyncio
async def test_mcp_endpoints_enabled() -> None:
    """Test MCP health and tools endpoints when MCP is enabled and initialized."""
    reset_mcp_service()
    settings = Settings(
        mcp_enabled=True,
        mcp_local_server_enabled=True,
        mcp_local_server_name="local",
        mcp_allowed_tools=["mcp.local.calculator", "mcp.local.text_stats"],
    )
    service = MCPService(settings=settings)
    await service.initialize()

    # Create app with custom settings override
    app = create_app()
    app.dependency_overrides[get_mcp_service] = lambda: service

    with TestClient(app) as test_client:
        health_resp = test_client.get("/api/v1/mcp/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["enabled"] is True
        assert health_data["servers"].get("local") == "connected"

        tools_resp = test_client.get("/api/v1/mcp/tools")
        assert tools_resp.status_code == 200
        tools_data = tools_resp.json()
        assert len(tools_data["servers"]) == 1
        server_entry = tools_data["servers"][0]
        assert server_entry["name"] == "local"
        assert server_entry["status"] == "connected"
        tool_names = [t["name"] for t in server_entry["tools"]]
        assert "mcp.local.calculator" in tool_names

    await service.shutdown()
    reset_mcp_service()
