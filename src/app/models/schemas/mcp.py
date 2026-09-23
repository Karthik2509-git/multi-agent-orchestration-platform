"""Schemas for Model Context Protocol (MCP) endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPToolItem(BaseModel):
    """Schema representing an exposed tool from an MCP server."""

    name: str = Field(description="Namespaced tool name (e.g. 'mcp.local.calculator')")
    description: str = Field(description="Description of what the tool does")
    source: str = Field(default="mcp", description="Tool source identifier")
    server: str = Field(description="Originating MCP server name")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema specifying acceptable parameters",
    )


class MCPServerInfo(BaseModel):
    """Information regarding a registered MCP server and its exposed tools."""

    name: str = Field(description="Server name")
    status: str = Field(description="Connection status (e.g. 'connected', 'disconnected')")
    tools: List[MCPToolItem] = Field(
        default_factory=list,
        description="List of approved tools exposed by this server",
    )


class MCPToolsListResponse(BaseModel):
    """Response payload for GET /api/v1/mcp/tools."""

    servers: List[MCPServerInfo] = Field(
        default_factory=list,
        description="List of configured MCP servers and their available tools",
    )


class MCPHealthResponse(BaseModel):
    """Response payload for GET /api/v1/mcp/health."""

    enabled: bool = Field(description="Whether MCP subsystem is enabled in settings")
    servers: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of server names to connection status",
    )
