"""MCP client wrapper providing lifecycle management and tool execution via MCP v2 Client."""

from typing import Any, Dict, List, Optional, Union

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

from src.app.core.logging import get_logger
from src.app.mcp.models import MCPToolDefinition
from src.app.tools.base import ToolResult

logger = get_logger(__name__)


class MCPClient:
    """High-level client wrapper delegating to official MCP SDK v2 Client."""

    def __init__(
        self,
        server_target: Union[MCPServer, Any],
        server_name: str = "local",
    ) -> None:
        self.server_target = server_target
        self.server_name = server_name
        self._client: Optional[Client] = None
        self._is_connected: bool = False

    @property
    def is_connected(self) -> bool:
        """Return whether client session is actively connected."""
        return self._is_connected and self._client is not None

    async def connect(self) -> None:
        """Establish session connection to the target MCP server."""
        if self.is_connected:
            return

        try:
            logger.debug("Connecting MCP client to server '%s'...", self.server_name)
            self._client = Client(self.server_target)
            await self._client.__aenter__()
            self._is_connected = True
            logger.info("Successfully connected MCP client to server '%s'", self.server_name)
        except Exception as e:
            self._is_connected = False
            self._client = None
            logger.error(
                "Failed to connect MCP client to server '%s': %s",
                self.server_name,
                str(e),
            )
            raise

    async def close(self) -> None:
        """Safely disconnect and tear down the MCP client session."""
        if self._client is not None:
            client = self._client
            self._client = None
            self._is_connected = False
            try:
                await client.__aexit__(None, None, None)
                logger.debug("Closed MCP client connection for server '%s'", self.server_name)
            except Exception as e:
                logger.warning(
                    "Error closing MCP client connection for '%s': %s",
                    self.server_name,
                    str(e),
                )

    async def list_tools(self) -> List[MCPToolDefinition]:
        """Discover tools exposed by the MCP server and return normalized definitions."""
        if not self.is_connected:
            await self.connect()

        assert self._client is not None
        try:
            logger.debug("Listing tools from MCP server '%s'...", self.server_name)
            response = await self._client.list_tools()
            raw_tools = getattr(response, "tools", [])

            normalized_tools: List[MCPToolDefinition] = []
            for tool in raw_tools:
                orig_name = str(tool.name)
                namespaced_name = f"mcp.{self.server_name}.{orig_name}"
                description = str(tool.description or "")
                input_schema = getattr(tool, "input_schema", {})
                if not isinstance(input_schema, dict):
                    input_schema = {}

                normalized = MCPToolDefinition(
                    name=namespaced_name,
                    server_name=self.server_name,
                    original_name=orig_name,
                    description=description,
                    input_schema=input_schema,
                    source="mcp",
                )
                normalized_tools.append(normalized)

            logger.info(
                "Discovered %d tools from MCP server '%s': %s",
                len(normalized_tools),
                self.server_name,
                [t.name for t in normalized_tools],
            )
            return normalized_tools
        except Exception as e:
            logger.error(
                "Failed to list tools from MCP server '%s': %s",
                self.server_name,
                str(e),
            )
            raise

    async def call_tool(self, original_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Invoke a tool on the MCP server and return a normalized ToolResult."""
        if not self.is_connected:
            try:
                await self.connect()
            except Exception as conn_err:
                return ToolResult(
                    success=False,
                    error=(
                        f"Failed to connect to MCP server '{self.server_name}': {str(conn_err)}"
                    ),
                )

        assert self._client is not None
        try:
            logger.debug(
                "Invoking tool '%s' on MCP server '%s' with args: %s",
                original_name,
                self.server_name,
                arguments,
            )
            response = await self._client.call_tool(
                name=original_name,
                arguments=arguments,
            )

            is_error = getattr(response, "is_error", False)
            content_blocks = getattr(response, "content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif hasattr(block, "text"):
                    text_parts.append(str(block.text))
                else:
                    text_parts.append(str(block))

            output_text = "\n".join(text_parts) if text_parts else ""

            if is_error:
                logger.warning(
                    "MCP server '%s' reported tool error for '%s': %s",
                    self.server_name,
                    original_name,
                    output_text,
                )
                return ToolResult(
                    success=False,
                    data=output_text,
                    error=output_text or f"MCP tool '{original_name}' reported error",
                )

            structured = getattr(response, "structured_content", None)
            return ToolResult(
                success=True,
                data=structured if structured is not None else output_text,
            )
        except Exception as exc:
            logger.error(
                "Exception calling MCP tool '%s' on server '%s': %s",
                original_name,
                self.server_name,
                str(exc),
            )
            return ToolResult(
                success=False,
                error=f"MCP invocation failed for '{original_name}': {str(exc)}",
            )
