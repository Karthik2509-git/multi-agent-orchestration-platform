# ADR 002: Model Context Protocol (MCP) as Extensible Tool Transport

## Status
Accepted

## Context
As agent ecosystems expand, integrating external tools and data sources requires standardizing how tools are described, discovered, and invoked. Hardcoding every tool directly into the application codebase couples the platform to specific third-party APIs and libraries. Anthropic's **Model Context Protocol (MCP)** provides an open standard for exposing tools, prompts, and resources over JSON-RPC transports. However, discarding native Python tools entirely in favor of out-of-process MCP servers introduces unnecessary latency and process overhead for core internal utilities.

## Decision
Implement a hybrid architecture where MCP **supplements rather than replaces** the internal `ToolRegistry`.
- Maintain an internal `ToolRegistry` as the single source of truth for agent execution.
- Retain native in-process tools (`CalculatorTool`, `SafeHTTPTool`, `KnowledgeSearchTool`) for high performance, zero subprocess overhead, and strict internal guardrails.
- Implement an `MCPClient` and `MCPServer` layer conforming to the official MCP specification.
- Use `MCPToolAdapter` to dynamically convert discovered MCP tools into the internal `BaseTool` interface, registering them with prefixed names (e.g., `mcp.local.<tool_name>`).

## Consequences

### Positive
- **Standardized Interoperability**: Any MCP-compliant server (local process or remote) can be connected without rewriting agent prompt wrappers.
- **Unified Tool Interface**: Specialist agents consume native and MCP tools through the exact same `BaseTool` abstraction and Pydantic schemas.
- **Process Isolation**: External or untrusted tool implementations run in separate processes without risking main backend stability.

### Negative / Trade-Offs
- Dual maintenance of native `BaseTool` models and MCP schema adapters.
- Subprocess/JSON-RPC overhead for external MCP tool invocations compared to direct Python function calls.

## Architectural Comparison
- **Pure Native Tool Registry**: Simpler, but requires writing bespoke integration code for every external tool, database, or API service.
- **Pure MCP Architecture**: Migrating all tools (including basic arithmetic) to out-of-process MCP servers would introduce unnecessary IPC latency and complex lifecycle management for simple built-in utilities.
