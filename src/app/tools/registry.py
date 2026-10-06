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
from src.app.tools.execution_context import ToolExecutionContext
from src.app.tools.guardrails import ToolGuardrails

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry for registering, discovering, and executing agent tools."""

    def __init__(self, guardrails: Optional[ToolGuardrails] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self.guardrails = guardrails or ToolGuardrails()

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

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """Execute a tool by name with provided arguments and run-scoped execution controls.

        Enforcement sequence:
        1. Validate tool exists.
        2. Check whether tool is disabled (run-scoped).
        3. Check global run budget and reserve attempt.
        4. Validate structured tool input via guardrails (underlying tool not called if invalid).
        5. Execute underlying tool.
        6. Validate structured tool output via guardrails.
        7. Record success or failure in run-scoped execution context (consecutive streak / totals).
        8. Record metrics and trace spans.
        """
        # 1. Validate tool exists
        tool = self.get(name)
        if not tool:
            available = list(self._tools.keys())
            logger.warning("Tool '%s' not found in registry", name)
            return ToolResult(
                success=False,
                error=f"Tool '{name}' is not registered. Available tools: {available}",
                error_category="validation_error",
            )

        # Fall back to a default single-call context if none provided (backward compatibility)
        ctx = ToolExecutionContext.ensure(context)

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

            # 2, 3, 4. Check disabled state, global budget, and reserve attempt
            blocked_result = await ctx.check_and_reserve(name)
            if blocked_result is not None:
                duration_s = time.perf_counter() - start_time
                duration_ms = round(duration_s * 1000, 2)
                err_category = blocked_result.error_category or "error"
                set_span_attributes(
                    span,
                    {
                        "status": "error",
                        "duration_ms": duration_ms,
                        "error_category": err_category,
                    },
                )
                record_tool_call(tool_name=name, tool_type=tool_type, status="error")
                record_tool_latency(
                    tool_name=name, tool_type=tool_type, duration_seconds=duration_s
                )
                record_tool_failure(tool_name=name, error=err_category)
                return blocked_result

            # 5. Input guardrails validation (underlying tool not called on failure)
            input_error = self.guardrails.validate_input(tool, arguments)
            if input_error is not None:
                duration_s = time.perf_counter() - start_time
                duration_ms = round(duration_s * 1000, 2)
                await ctx.record_execution_result(name, success=False)
                set_span_attributes(
                    span,
                    {
                        "status": "error",
                        "duration_ms": duration_ms,
                        "error_category": "validation_error",
                    },
                )
                record_tool_call(tool_name=name, tool_type=tool_type, status="error")
                record_tool_latency(
                    tool_name=name, tool_type=tool_type, duration_seconds=duration_s
                )
                record_tool_failure(tool_name=name, error="validation_error")
                return input_error

            # 6. Execute the underlying tool
            try:
                logger.info("Executing tool '%s' with arguments: %s", name, arguments)
                result = await tool.execute(**arguments)
                duration_s = time.perf_counter() - start_time
                duration_ms = round(duration_s * 1000, 2)

                # 7. Output guardrails validation
                if result.success:
                    result = self.guardrails.validate_output(name, result)

                # 8. Record success or failure in execution context
                await ctx.record_execution_result(name, success=result.success)

                # 9. Record observability
                tool_status = "success" if result.success else "error"
                error_cat = result.error_category or (
                    "execution_error" if not result.success else None
                )
                span_updates: Dict[str, Any] = {
                    "status": tool_status,
                    "duration_ms": duration_ms,
                }
                if error_cat:
                    span_updates["error_category"] = error_cat
                set_span_attributes(span, span_updates)

                record_tool_call(tool_name=name, tool_type=tool_type, status=tool_status)
                record_tool_latency(
                    tool_name=name, tool_type=tool_type, duration_seconds=duration_s
                )
                if not result.success:
                    record_tool_failure(
                        tool_name=name,
                        error=result.error_category or result.error or "execution_error",
                    )
                return result
            except Exception as e:
                duration_s = time.perf_counter() - start_time
                duration_ms = round(duration_s * 1000, 2)
                logger.error("Unhandled exception executing tool '%s': %s", name, str(e))
                await ctx.record_execution_result(name, success=False)
                record_span_error(span, e)
                set_span_attributes(
                    span,
                    {
                        "status": "error",
                        "duration_ms": duration_ms,
                        "error_category": "execution_error",
                    },
                )
                record_tool_call(tool_name=name, tool_type=tool_type, status="error")
                record_tool_latency(
                    tool_name=name, tool_type=tool_type, duration_seconds=duration_s
                )
                record_tool_failure(tool_name=name, error=e)
                return ToolResult(
                    success=False,
                    error=f"Internal error executing tool '{name}': {str(e)}",
                    error_category="execution_error",
                )
