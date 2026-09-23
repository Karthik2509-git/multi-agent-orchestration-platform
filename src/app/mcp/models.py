"""Internal normalized data models and contracts for the MCP subsystem."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MCPToolDefinition(BaseModel):
    """Normalized metadata for a tool discovered from an MCP server."""

    name: str = Field(description="Namespaced unique tool identifier (e.g. 'mcp.local.calculator')")
    server_name: str = Field(description="Name of the MCP server exposing this tool")
    original_name: str = Field(description="Original tool name as declared by the MCP server")
    description: str = Field(description="Documentation and usage instructions for the tool")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema specification for tool input arguments",
    )
    source: str = Field(default="mcp", description="Origin source identifier")


class MCPServerStatus(BaseModel):
    """Connection and health status of a registered MCP server."""

    server_name: str = Field(description="Identifier of the MCP server")
    status: str = Field(
        default="connected",
        description="Status string: 'connected', 'disconnected', 'error'",
    )
    tools_count: int = Field(default=0, description="Number of discovered and approved tools")
    error: Optional[str] = Field(default=None, description="Diagnostic error message if failure")
