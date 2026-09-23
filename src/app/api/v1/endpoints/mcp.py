"""Model Context Protocol (MCP) tool discovery and health endpoints."""

from fastapi import APIRouter, Depends, status

from src.app.core.logging import get_logger
from src.app.models.schemas.mcp import MCPHealthResponse, MCPToolsListResponse
from src.app.services.mcp_service import MCPService, get_mcp_service

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=MCPHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="MCP Subsystem Health Check",
    description="Returns configuration and connection status of the MCP subsystem and servers.",
)
async def get_mcp_health(
    mcp_service: MCPService = Depends(get_mcp_service),
) -> MCPHealthResponse:
    """Return MCP subsystem health status."""
    return mcp_service.get_health()


@router.get(
    "/tools",
    response_model=MCPToolsListResponse,
    status_code=status.HTTP_200_OK,
    summary="List MCP Discovered Tools",
    description="Returns all approved tools discovered across configured MCP servers.",
)
async def list_mcp_tools(
    mcp_service: MCPService = Depends(get_mcp_service),
) -> MCPToolsListResponse:
    """Return discovered tools grouped by configured MCP servers."""
    return mcp_service.get_tools_list()
