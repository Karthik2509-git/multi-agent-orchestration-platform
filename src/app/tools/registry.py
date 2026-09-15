"""Tool registry managing tool lifecycle, schemas, and execution."""

from typing import Any, Dict, List, Optional

from src.app.core.logging import get_logger
from src.app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry for registering, discovering, and executing agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool '%s' in registry", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Export OpenAI-compatible function calling schemas for all registered tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with provided arguments."""
        tool = self.get(name)
        if not tool:
            available = list(self._tools.keys())
            logger.warning("Tool '%s' not found in registry", name)
            return ToolResult(
                success=False,
                error=f"Tool '{name}' is not registered. Available tools: {available}",
            )

        try:
            logger.info("Executing tool '%s' with arguments: %s", name, arguments)
            return await tool.execute(**arguments)
        except Exception as e:
            logger.error("Unhandled exception executing tool '%s': %s", name, str(e))
            return ToolResult(
                success=False,
                error=f"Internal error executing tool '{name}': {str(e)}",
            )
