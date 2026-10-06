import time
from typing import Any, Dict, List, Optional

from src.app.core.logging import get_logger
from src.app.observability import (
    record_span_error,
    record_tool_call,
    record_tool_failure,
    record_tool_latency,
    set_span_attributes,
    trace_tool_execution,
)
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

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered by name."""
        return name in self._tools

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator to check tool existence."""
        return name in self._tools

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

        tool_type = (
            "mcp"
            if (getattr(tool, "source", None) == "mcp" or name.startswith("mcp."))
            else "native"
        )
        extra_attrs: Dict[str, Any] = {}
        if hasattr(tool, "server_name"):
            extra_attrs["mcp.server_name"] = getattr(tool, "server_name")
        if hasattr(tool, "original_name"):
            extra_attrs["mcp.original_name"] = getattr(tool, "original_name")

        async with trace_tool_execution(
            tool_name=name,
            tool_type=tool_type,
            extra_attributes=extra_attrs,
        ) as span:
            start_time = time.perf_counter()
            try:
                logger.info("Executing tool '%s' with arguments: %s", name, arguments)
                result = await tool.execute(**arguments)
                duration_s = time.perf_counter() - start_time
                duration_ms = round(duration_s * 1000, 2)
                set_span_attributes(
                    span,
                    {
                        "status": "success" if result.success else "error",
                        "duration_ms": duration_ms,
                    },
                )
                tool_status = "success" if result.success else "error"
                record_tool_call(tool_name=name, tool_type=tool_type, status=tool_status)
                record_tool_latency(
                    tool_name=name, tool_type=tool_type, duration_seconds=duration_s
                )
                if not result.success:
                    set_span_attributes(span, {"error_category": "tool_failure"})
                    record_tool_failure(tool_name=name, error=result.error or "execution_error")
                return result
            except Exception as e:
                duration_s = time.perf_counter() - start_time
                duration_ms = round(duration_s * 1000, 2)
                logger.error("Unhandled exception executing tool '%s': %s", name, str(e))
                record_span_error(span, e)
                set_span_attributes(span, {"status": "error", "duration_ms": duration_ms})
                record_tool_call(tool_name=name, tool_type=tool_type, status="error")
                record_tool_latency(
                    tool_name=name, tool_type=tool_type, duration_seconds=duration_s
                )
                record_tool_failure(tool_name=name, error=e)
                return ToolResult(
                    success=False,
                    error=f"Internal error executing tool '{name}': {str(e)}",
                )
