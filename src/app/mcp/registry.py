"""MCP tool registry managing discovered tools, allowlisting, and ToolRegistry integration."""

from typing import Dict, List, Optional, Sequence, Set

from src.app.core.logging import get_logger
from src.app.mcp.adapters import MCPToolAdapter
from src.app.mcp.client import MCPClient
from src.app.mcp.models import MCPServerStatus, MCPToolDefinition
from src.app.tools.registry import ToolRegistry

logger = get_logger(__name__)


class MCPToolRegistry:
    """Registry responsible for discovering, filtering, and adapting MCP tools."""

    def __init__(self, allowed_tools: Optional[Sequence[str]] = None) -> None:
        self._servers: Dict[str, MCPClient] = {}
        self._discovered_definitions: Dict[str, MCPToolDefinition] = {}
        self._adapters: Dict[str, MCPToolAdapter] = {}
        self._allowed_tools: Optional[Set[str]] = (
            {t.lower() for t in allowed_tools} if allowed_tools is not None else None
        )

    def register_server(self, name: str, client: MCPClient) -> None:
        """Register an MCP server client with the registry."""
        if name in self._servers:
            logger.warning("Replacing existing MCP server client for '%s'", name)
        self._servers[name] = client
        logger.info("Registered MCP server '%s' in MCPToolRegistry", name)

    def unregister_server(self, name: str) -> Optional[MCPClient]:
        """Unregister an MCP server client and remove its tools."""
        client = self._servers.pop(name, None)
        # Remove associated definitions and adapters
        to_remove_defs = [
            k for k, v in self._discovered_definitions.items() if v.server_name == name
        ]
        for k in to_remove_defs:
            self._discovered_definitions.pop(k, None)

        to_remove_adapters = [k for k, v in self._adapters.items() if v.server_name == name]
        for k in to_remove_adapters:
            self._adapters.pop(k, None)

        logger.info("Unregistered MCP server '%s' and removed associated tools", name)
        return client

    def get_server(self, name: str) -> Optional[MCPClient]:
        """Get registered MCP server client by name."""
        return self._servers.get(name)

    def list_servers(self) -> List[str]:
        """Return names of all registered MCP servers."""
        return list(self._servers.keys())

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check whether a namespaced tool name is permitted by the allowlist."""
        if self._allowed_tools is None:
            return True
        return tool_name.lower() in self._allowed_tools

    async def discover_tools(self) -> List[MCPToolDefinition]:
        """Discover tools from all connected MCP servers and build adapters."""
        all_definitions: List[MCPToolDefinition] = []

        for server_name, client in self._servers.items():
            try:
                tools = await client.list_tools()
                for defn in tools:
                    # Filter through allowlist
                    if not self.is_tool_allowed(defn.name):
                        logger.debug(
                            "Tool '%s' discovered from server '%s' is not in allowlist; skipping.",
                            defn.name,
                            server_name,
                        )
                        continue

                    self._discovered_definitions[defn.name] = defn
                    adapter = MCPToolAdapter(definition=defn, client=client)
                    self._adapters[defn.name] = adapter
                    all_definitions.append(defn)
            except Exception as e:
                logger.error(
                    "Failed tool discovery for MCP server '%s': %s",
                    server_name,
                    str(e),
                )

        logger.info(
            "MCPToolRegistry finished discovery. Approved and adapted tools: %s",
            list(self._adapters.keys()),
        )
        return all_definitions

    def get_tool_definitions(self) -> List[MCPToolDefinition]:
        """Return all approved discovered MCP tool definitions."""
        return list(self._discovered_definitions.values())

    def get_tool_adapters(self) -> List[MCPToolAdapter]:
        """Return all approved MCPToolAdapter instances."""
        return list(self._adapters.values())

    def get_adapter(self, name: str) -> Optional[MCPToolAdapter]:
        """Retrieve a specific MCPToolAdapter by namespaced name."""
        return self._adapters.get(name)

    def register_into_tool_registry(self, target_registry: ToolRegistry) -> List[str]:
        """Register all approved MCP tool adapters into the application's central ToolRegistry.

        Guarantees:
        - Prevents collisions with existing native tools (rejects registration and logs error).
        - Preserves existing tools without silent overwriting.
        """
        registered_names: List[str] = []
        for name, adapter in self._adapters.items():
            existing = target_registry.get(name)
            if existing is not None:
                # Collision detected with an existing tool
                logger.error(
                    "Cannot register MCP tool '%s': collision detected with existing tool (%s). "
                    "Skipping to prevent overwrite.",
                    name,
                    type(existing).__name__,
                )
                continue

            target_registry.register(adapter)
            registered_names.append(name)
            logger.debug("Injected MCP tool adapter '%s' into ToolRegistry", name)

        logger.info(
            "Injected %d MCP tools into application ToolRegistry: %s",
            len(registered_names),
            registered_names,
        )
        return registered_names

    def get_server_statuses(self) -> Dict[str, MCPServerStatus]:
        """Return status and tool counts for all registered MCP servers."""
        statuses: Dict[str, MCPServerStatus] = {}
        for server_name, client in self._servers.items():
            tools_count = sum(
                1
                for defn in self._discovered_definitions.values()
                if defn.server_name == server_name
            )
            status = "connected" if client.is_connected else "disconnected"
            statuses[server_name] = MCPServerStatus(
                server_name=server_name,
                status=status,
                tools_count=tools_count,
            )
        return statuses
