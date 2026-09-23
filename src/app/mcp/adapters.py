"""Adapter converting discovered MCP tools into the platform's BaseTool interface."""

from typing import Any, Dict

from src.app.core.logging import get_logger
from src.app.mcp.client import MCPClient
from src.app.mcp.models import MCPToolDefinition
from src.app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class MCPToolAdapter(BaseTool):
    """Drop-in adapter making an MCP-discovered tool compatible with BaseTool and ToolRegistry."""

    def __init__(
        self,
        definition: MCPToolDefinition,
        client: MCPClient,
    ) -> None:
        self.name = definition.name
        self.description = definition.description
        self.parameters_schema = definition.input_schema
        self.original_name = definition.original_name
        self.server_name = definition.server_name
        self.source = definition.source
        self.client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool via the bound MCP client."""
        logger.debug(
            "MCPToolAdapter executing '%s' (server: %s, original: %s) with kwargs: %s",
            self.name,
            self.server_name,
            self.original_name,
            kwargs,
        )
        return await self.client.call_tool(
            original_name=self.original_name,
            arguments=kwargs,
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool schema into standard OpenAI function calling format."""
        schema = super().to_openai_schema()
        # Guarantee parameters is an object schema
        if not schema.get("parameters"):
            schema["parameters"] = {"type": "object", "properties": {}}
        return schema
