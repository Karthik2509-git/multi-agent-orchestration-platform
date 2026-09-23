"""Service managing MCP subsystem lifecycle, discovery, and ToolRegistry integration."""

from typing import Dict, Optional

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.mcp.client import MCPClient
from src.app.mcp.models import MCPServerStatus
from src.app.mcp.registry import MCPToolRegistry
from src.app.mcp.servers.local_tools import create_local_mcp_server
from src.app.models.schemas.mcp import (
    MCPHealthResponse,
    MCPServerInfo,
    MCPToolItem,
    MCPToolsListResponse,
)
from src.app.tools.registry import ToolRegistry

logger = get_logger(__name__)


class MCPService:
    """Coordinates MCP servers, client connections, tool discovery, and tool registry injection."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = MCPToolRegistry(allowed_tools=settings.mcp_allowed_tools)
        self._is_initialized = False

    @property
    def is_enabled(self) -> bool:
        """Return whether MCP is enabled in settings."""
        return self.settings.mcp_enabled

    async def initialize(self) -> None:
        """Initialize configured MCP servers and discover approved tools."""
        if not self.is_enabled:
            logger.info("MCP subsystem is disabled (MCP_ENABLED=false)")
            return

        logger.info("Initializing MCP subsystem...")

        # Setup local in-process MCP server if enabled
        if self.settings.mcp_local_server_enabled:
            server_name = self.settings.mcp_local_server_name
            try:
                local_server = create_local_mcp_server(name=server_name)
                client = MCPClient(
                    server_target=local_server,
                    server_name=server_name,
                )
                await client.connect()
                self.registry.register_server(name=server_name, client=client)
                logger.info("Configured and connected local MCP server '%s'", server_name)
            except Exception as e:
                logger.error("Failed to initialize local MCP server '%s': %s", server_name, str(e))

        # Perform initial tool discovery across all configured servers
        try:
            discovered = await self.registry.discover_tools()
            logger.info(
                "MCP subsystem initialized with %d approved tools across %d servers",
                len(discovered),
                len(self.registry.list_servers()),
            )
        except Exception as exc:
            logger.error("Error during initial MCP tool discovery: %s", str(exc))

        self._is_initialized = True

    async def shutdown(self) -> None:
        """Gracefully disconnect all active MCP clients and clean up resources."""
        logger.info("Shutting down MCP subsystem...")
        for server_name in self.registry.list_servers():
            client = self.registry.get_server(server_name)
            if client is not None:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning("Error closing MCP client '%s': %s", server_name, str(e))
        self._is_initialized = False
        logger.info("MCP subsystem shutdown complete")

    def register_tools_into(self, target_registry: ToolRegistry) -> None:
        """Inject approved MCP tool adapters into the application's ToolRegistry."""
        if not self.is_enabled:
            return
        self.registry.register_into_tool_registry(target_registry)

    def get_health(self) -> MCPHealthResponse:
        """Generate lightweight health status for the MCP subsystem."""
        server_statuses: Dict[str, MCPServerStatus] = self.registry.get_server_statuses()
        servers_map = {name: status.status for name, status in server_statuses.items()}

        return MCPHealthResponse(
            enabled=self.is_enabled,
            servers=servers_map,
        )

    def get_tools_list(self) -> MCPToolsListResponse:
        """Generate list of configured servers and their exposed tools."""
        definitions = self.registry.get_tool_definitions()
        server_statuses = self.registry.get_server_statuses()

        servers_info: list[MCPServerInfo] = []

        # If servers are registered
        for server_name in self.registry.list_servers():
            status_obj = server_statuses.get(server_name)
            status_str = status_obj.status if status_obj else "unknown"

            server_tools = [
                MCPToolItem(
                    name=defn.name,
                    description=defn.description,
                    source=defn.source,
                    server=defn.server_name,
                    parameters=defn.input_schema,
                )
                for defn in definitions
                if defn.server_name == server_name
            ]

            servers_info.append(
                MCPServerInfo(
                    name=server_name,
                    status=status_str,
                    tools=server_tools,
                )
            )

        return MCPToolsListResponse(servers=servers_info)


# Singleton instance for application lifecycle injection
_mcp_service_instance: Optional[MCPService] = None


def get_mcp_service(settings: Optional[Settings] = None) -> MCPService:
    """Retrieve or create the global MCPService instance."""
    global _mcp_service_instance
    if _mcp_service_instance is None:
        current_settings = settings or get_settings()
        _mcp_service_instance = MCPService(settings=current_settings)
    return _mcp_service_instance


def reset_mcp_service() -> None:
    """Reset the singleton instance (primarily for testing)."""
    global _mcp_service_instance
    _mcp_service_instance = None
