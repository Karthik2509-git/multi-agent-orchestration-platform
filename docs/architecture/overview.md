# Architecture Overview

The **Multi-Agent AI Orchestration Platform** is a production-oriented, modular backend system engineered to coordinate specialized autonomous agents, standardized tool ecosystems, retrieval-augmented generation (RAG), persistent memory, human oversight, and execution replay.

The platform is designed around strict separation of concerns, defensive runtime controls, and deterministic graph state management.

---

## 1. High-Level System Architecture

The following diagram illustrates the active runtime boundaries and data flow across all subsystems:

```mermaid
graph TD
    Client["Client / External Consumer"] -->|HTTP / JSON| API["FastAPI REST API Layer<br/>(21 Paths / 24 Operations)"]

    subgraph Core Services
        API --> HealthService["HealthService<br/>(Liveness & Self-Healing Readiness)"]
        API --> AgentService["AgentService<br/>(Single Agent + Tool Registry)"]
        API --> OrchestrationService["OrchestrationService<br/>(LangGraph Coordinator)"]
        API --> RAGService["RAGService<br/>(Hybrid Ingestion & Search)"]
        API --> MemoryService["MemoryService<br/>(Chroma Semantic Store)"]
        API --> MCPService["MCPService<br/>(Server & Client Adapter)"]
        API --> ReplayService["ReplayService<br/>(Thread Fork & State Override)"]
    end

    subgraph LangGraph Orchestration Engine
        OrchestrationService --> Supervisor["Supervisor Agent<br/>(Intent Classification & Plan)"]
        Supervisor -->|Route| ResearchAgent["Research Agent<br/>(Web & Doc Search)"]
        Supervisor -->|Route| DataAgent["Data Agent<br/>(Arithmetic & Stats)"]
        Supervisor -->|Route| CodeAgent["Code Agent<br/>(Static Code Synthesis)"]
        Supervisor -->|Synthesize| FinalAgent["Final Agent<br/>(Answer Aggregation)"]

        Supervisor -.->|Sensitive / Low Confidence| ApprovalGate["ApprovalGate Node<br/>(LangGraph interrupt)"]
    end

    subgraph Tool & Extension Ecosystem
        ResearchAgent --> ToolRegistry["Tool Registry<br/>(Guarded Execution)"]
        DataAgent --> ToolRegistry

        ToolRegistry --> NativeTools["Native Tools<br/>• Calculator (AST)<br/>• HTTP GET (SSRF-Hardened)"]
        ToolRegistry --> RAGTool["Knowledge Search Tool<br/>(Hybrid RAG)"]
        ToolRegistry --> MCPAdapter["MCP Tool Adapter<br/>(Local MCP Tools)"]
    end

    subgraph State, Memory & Persistence
        ApprovalGate -.-> Checkpointer["LangGraph Checkpointer<br/>(AsyncPostgresSaver / MemorySaver)"]
        OrchestrationService --> Checkpointer
        OrchestrationService --> LongTermMemory["Long-Term Semantic Memory<br/>(Chroma: agent_memory)"]
        RAGService --> VectorStore["Vector Store<br/>(Chroma: knowledge_base)"]
    end

    subgraph Observability & Guardrails
        API -.-> OTel["OpenTelemetry Tracing<br/>(Custom Spans & Context Propagation)"]
        ToolRegistry -.-> Guardrails["Tool Guardrails<br/>(Budget: 10 calls, Failures: 3, Payload limits)"]
        OrchestrationService -.-> CostTracker["Cost Tracker<br/>(Per-Model Token Accounting)"]
    end
```

---

## 2. Core Architectural Layers

### 2.1 API & Ingress Layer (`src/app/api/`)
- Built with **FastAPI** providing typed request/response contracts via **Pydantic v2**.
- Exposes 21 distinct paths (24 HTTP operations) grouped into functional domains: Health, Agent, Orchestration, HITL, Knowledge, Memory, MCP, and LLM Provider Status.
- Structured with centralized error handlers, CORS middleware, and dependency injection.

### 2.2 Orchestration Layer (`src/app/orchestration/`)
- Implemented using **LangGraph StateGraph**.
- State is modeled as a strongly-typed `OrchestrationState` holding task specifications, execution history, accumulated context, agent usage, token metrics, cost tracking, and human-in-the-loop pending approval flags.
- Driven by a **Supervisor Agent** that dynamically evaluates intermediate results, plans subsequent steps, routes to specialized agents, or delegates to `FinalAgent` for synthesis.
- Enforces an execution loop ceiling (`MAX_ITERATIONS = 10`) to eliminate runaway recursion.

### 2.3 Specialized Agents (`src/app/agents/`)
- **`SupervisorAgent`**: Deconstructs user objectives, validates previous step outcomes, injects relevant long-term memory context, and selects next specialist via a strict allowlist (`research`, `data`, `code`, `final`, `__end__`).
- **`ResearchAgent`**: Retrieves external data using safe HTTP requests, queries internal documentation via RAG, and leverages MCP tools.
- **`DataAgent`**: Performs deterministic arithmetic calculations and statistical computations via AST-based tools.
- **`CodeAgent`**: Generates software architectures, diffs, and scripts with a **zero-execution guarantee** (code is analyzed and authored, never executed on host).
- **`FinalAgent`**: Aggregates multi-agent step outputs, reconciles contradictions, and formats the user response.

### 2.4 Tool Registry & Guardrail Ecosystem (`src/app/tools/`)
- Unified `ToolRegistry` managing native tools, RAG search, and Model Context Protocol (MCP) integrations.
- Defensive execution wrapper enforcing:
  - **Budget Ceiling**: Maximum 10 tool calls per execution run.
  - **Consecutive-Failure Tool Disablement**: Tracks failures per tool, resets counter on success, and disables the tool for the remainder of the run if it reaches 3 consecutive failures.
  - **Payload Limits**: 64 KB maximum tool input; 1 MB maximum tool output.
  - **Failure Isolation**: Unhandled tool exceptions are caught, logged, and returned as structured errors to the LLM without terminating the graph execution.

### 2.5 Retrieval-Augmented Generation (`src/app/rag/`)
- Dual-mode retrieval combining:
  - **Dense Vector Search**: Chroma vector database using local ONNX embeddings (`all-MiniLM-L6-v2`) or deterministic mock embeddings for offline testing.
  - **Sparse Lexical Search**: BM25 keyword matching.
- **Reciprocal Rank Fusion (RRF)** combines dense and sparse ranks with a balance coefficient ($\alpha = 0.6$) and rank constant ($k = 60$).
- Supports `.txt`, `.md`, and `.pdf` ingestion with recursive character chunking and metadata preservation.

### 2.6 Dual Memory System (`src/app/memory/`)
- **Working Memory**: LangGraph checkpointing storing execution graphs, node outputs, and thread checkpoints. Backed by PostgreSQL (`AsyncPostgresSaver`) in production or in-memory (`MemorySaver`) for ephemeral testing.
- **Long-Term Semantic Memory**: Chroma-backed semantic memory (`agent_memory` collection) partitioned by scope (`user`, `agent`, `session`, `global`). Features automated entity extraction, cosine deduplication ($>0.85$), TTL expiry cleanup, and composite ranking ($0.65\text{sim} + 0.25\text{imp} + 0.10\text{recency}$).

### 2.7 Human-in-the-Loop (HITL) & Replay (`src/app/hitl/`, `src/app/replay/`)
- Graph interrupts triggered via `interrupt()` when sensitive tool patterns occur or supervisor confidence falls below $0.65$.
- Thread execution state is persisted to the checkpointer, allowing asynchronous inspection via `GET /api/v1/hitl/pending/{thread_id}` and resumption via `POST /api/v1/hitl/resume/{thread_id}`.
- Replay engine (`ReplayService`) enables time-travel debugging: re-executing runs from arbitrary checkpoint nodes, overriding inputs, or injecting mock tool responses with safety guardrails.

### 2.8 Observability & Reliability (`src/app/observability/`, `src/app/services/health.py`)
- **OpenTelemetry Tracing**: Contextual span tracking across API endpoints, graph nodes, agent reasoning, tool execution, and LLM calls.
- **Metrics Collection**: Process-local metrics recording latency, token volume, error rates, and cache hits.
- **Cost Tracking**: Exact token calculation and model-specific pricing attribution.
- **Self-Healing Readiness**: Asynchronous connection pool initialization in `HealthService` ensures transient database unavailability during container startup recovers automatically without restarting the application.
