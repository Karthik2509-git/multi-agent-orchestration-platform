"""Service coordinating agent initialization, tool registration, and execution."""

from typing import Optional

from src.app.agents.tool_calling_agent import ToolCallingAgent
from src.app.core.config import Settings
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.llm.factory import get_llm_provider
from src.app.models.schemas.agent import AgentRunResponse
from src.app.services.mcp_service import MCPService, get_mcp_service
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool
from src.app.tools.registry import ToolRegistry

logger = get_logger(__name__)


def build_default_tool_registry(
    settings: Settings,
    mcp_service: Optional[MCPService] = None,
) -> ToolRegistry:
    """Construct standard ToolRegistry with Calculator, SafeHTTP, and approved MCP tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        SafeHTTPGetTool(
            allowed_domains=settings.allowed_http_domains,
            timeout=settings.tool_http_timeout,
            max_size_bytes=settings.tool_http_max_size_bytes,
        )
    )

    # Register approved MCP tools if MCP is active
    active_mcp = mcp_service or (get_mcp_service(settings) if settings.mcp_enabled else None)
    if active_mcp is not None:
        active_mcp.register_tools_into(registry)

    return registry


async def run_agent_task(
    task: str,
    settings: Settings,
    provider: Optional[LLMProvider] = None,
    registry: Optional[ToolRegistry] = None,
) -> AgentRunResponse:
    """Coordinate end-to-end agent task execution."""
    # Resolve provider
    if provider is None:
        provider = get_llm_provider(settings)

    # Resolve tool registry
    if registry is None:
        registry = build_default_tool_registry(settings)

    agent = ToolCallingAgent(
        provider=provider,
        registry=registry,
        max_iterations=settings.agent_max_iterations,
    )

    return await agent.run(task)
