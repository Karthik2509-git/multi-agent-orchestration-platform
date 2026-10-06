# Component & Subsystem Map

This document details the responsibilities, dependencies, and boundaries of every core subsystem in the platform.

---

## 1. Directory Structure & Responsibilities

```
src/app/
├── api/                   # REST API ingress, routing, and endpoint definitions
│   └── v1/
│       ├── api.py         # Root v1 APIRouter registering all domain routers
│       └── endpoints/     # Domain-specific route controllers
├── agents/                # Autonomous agent logic and specialist personas
├── orchestration/         # LangGraph workflows, state definitions, and graph topology
├── llm/                   # Multi-provider LLM abstraction layer
├── tools/                 # Tool implementations, registry, and execution guards
├── mcp/                   # Model Context Protocol client, server, and adapters
├── rag/                   # Hybrid retrieval-augmented generation pipeline
├── memory/                # Working memory checkpointer and long-term semantic memory
├── hitl/                  # Human-in-the-loop policies, gating, and state inspection
├── observability/         # OpenTelemetry tracing, metrics collection, and cost tracking
├── evaluation/            # Offline evaluation framework, metrics, and benchmark runners
├── replay/                # Time-travel checkpoint replay, fork, and override engine
├── services/              # Stateless business logic and cross-cutting service layer
├── core/                  # Global application configuration, logging, and settings
├── models/                # Pydantic schemas, data contracts, and domain types
├── db/                    # Database session management and connection utilities
└── main.py                # ASGI application factory, lifespan hooks, CORS, and root routes
```

---

## 2. Subsystem Details

### 2.1 API Ingress (`src/app/api/`)
- **Responsibility**: Expose HTTP endpoints, parse inputs into Pydantic models, handle HTTP exceptions, and format JSON responses.
- **Key Modules**:
  - `v1/endpoints/health.py`: Liveness (`/health`) and detailed readiness (`/api/v1/health`).
  - `v1/endpoints/agent.py`: Direct single-agent runs (`/api/v1/agent/run`).
  - `v1/endpoints/orchestration.py`: LangGraph runs (`/api/v1/orchestration/run`) and replay (`/api/v1/orchestration/replay`).
  - `v1/endpoints/hitl.py`: State inspection (`/pending/{thread_id}`) and approval (`/resume/{thread_id}`).
  - `v1/endpoints/knowledge.py`: Document ingestion, hybrid search, RAG query.
  - `v1/endpoints/memory.py`: Semantic memory CRUD, search, and consolidation.
  - `v1/endpoints/mcp.py`: MCP server health and registered tools discovery.
  - `v1/endpoints/llm.py`: Active provider status and supported model list.

### 2.2 Agent Personas (`src/app/agents/`)
- **Responsibility**: Implement specialized LLM agent logic using standardized interfaces.
- **Key Classes**:
  - `BaseSpecializedAgent`: Abstract contract defining `run(task, context) -> AgentResult`.
  - `SupervisorAgent`: Dynamic intent classifier, routing logic, allowlist validation.
  - `ResearchAgent`: Information gathering agent equipped with search and HTTP tools.
  - `DataAgent`: Quantitative processing agent equipped with AST calculator.
  - `CodeAgent`: Static code analysis and generation with zero-execution safety guarantee.
  - `FinalAgent`: Aggregator that synthesizes multi-step results into final answers.
  - `ToolCallingAgent`: Single-agent tool execution loop for direct endpoint requests.

### 2.3 Orchestration Engine (`src/app/orchestration/`)
- **Responsibility**: Coordinate state transitions, graph cycles, and checkpointer persistence.
- **Key Classes**:
  - `OrchestrationState`: Typed dictionary tracking `task`, `thread_id`, `plan`, `current_step`, `agent_history`, `context`, `agents_used`, `answer`, `status`, `approval_status`, `tool_call_count`, `total_cost_usd`.
  - `graph.py`: Compiles the `StateGraph` linking nodes (`supervisor`, `research`, `data`, `code`, `final`, `approval_gate`).

### 2.4 LLM Abstraction Layer (`src/app/llm/`)
- **Responsibility**: Provide provider-agnostic text and tool-calling interfaces.
- **Key Classes**:
  - `LLMProvider`: Abstract base class defining `agenerate()` and `agenerate_with_tools()`.
  - `LLMFactory`: Factory creating providers dynamically from application configuration.
  - Providers: `OpenRouterProvider`, `GeminiProvider`, `GroqProvider`, `MockLLMProvider`, `OpenAIProvider`.

### 2.5 Tool Registry & Guardrails (`src/app/tools/`)
- **Responsibility**: Tool discovery, schema generation, execution boundaries, and safety.
- **Key Classes**:
  - `BaseTool`: Typed contract with parameter validation via Pydantic schemas.
  - `ToolRegistry`: Thread-safe registry for registering and resolving tools.
  - `CalculatorTool`: Safe AST-based math evaluator (blocks `eval`, `exec`, imports, and high power operations).
  - `SafeHTTPTool`: SSRF-hardened HTTP client blocking private/loopback/multicast IPs and unapproved domains.
  - `ToolGuardrail`: Enforces call budgets (10 calls/run), consecutive-failure tool disablement (3 failures disables tool for remainder of run), and input/output payload limits.

### 2.6 Model Context Protocol (`src/app/mcp/`)
- **Responsibility**: Standardized external tool discovery and invocation via official MCP SDK.
- **Key Classes**:
  - `MCPServer`: In-process reference MCP server exposing `mcp.local.calculator` and `mcp.local.text_stats`.
  - `MCPClient`: Client managing communication with configured MCP servers.
  - `MCPToolAdapter`: Adapts external MCP tools to the internal `BaseTool` interface.

### 2.7 Retrieval-Augmented Generation (`src/app/rag/`)
- **Responsibility**: Document parsing, chunking, indexing, and hybrid retrieval.
- **Key Classes**:
  - `DocumentParser`: Extracts text from `.txt`, `.md`, and `.pdf` files.
  - `RecursiveChunker`: Semantic chunking with configurable overlap.
  - `ChromaVectorStore`: Persistent vector store using Chroma DB.
  - `HybridRetriever`: Combines dense semantic embeddings with BM25 keyword search using Reciprocal Rank Fusion ($\alpha=0.6$).

### 2.8 Memory Subsystem (`src/app/memory/`)
- **Responsibility**: Dual-tier storage for working memory and long-term knowledge.
- **Key Classes**:
  - Working Memory: Backed by `AsyncPostgresSaver` in production and `MemorySaver` in development.
  - `MemoryStore`: Semantic store backed by Chroma `agent_memory` collection.
  - `MemoryExtractor`: LLM-assisted or heuristic extractor filtering facts, preferences, and constraints.
  - `MemoryConsolidator`: Sweeper merging duplicate vectors ($>0.85$ cosine similarity) and purging expired TTL items.

### 2.9 Observability & Governance (`src/app/observability/`, `src/app/hitl/`, `src/app/evaluation/`, `src/app/replay/`)
- **Responsibility**: Tracing, metrics, safety gates, evaluation, and time-travel replay.
- **Key Classes**:
  - `tracer`: OpenTelemetry tracer recording hierarchical operation spans.
  - `SystemMetrics`: In-memory counter and histogram collector.
  - `CostCalculator`: Per-token pricing calculator supporting multiple models.
  - `ApprovalGate`: Evaluates actions against safety policies and executes `interrupt()`.
  - `ReplayService`: Restores thread state from checkpointer and forks execution.
  - `EvaluationRunner`: Evaluates agent trajectories against ground truth benchmarks.
