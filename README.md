# Multi-Agent AI Orchestration Platform

A production-grade, modular backend system engineered to coordinate specialized autonomous agents, standardized tool ecosystems, hybrid retrieval (RAG), persistent memory, human oversight, and execution replay.

---

## 🏛️ System Overview

The **Multi-Agent AI Orchestration Platform** provides an enterprise-ready reference architecture for complex, multi-agent AI systems. Rather than relying on simple linear prompt chains or unconstrained autonomous loops, the platform employs **LangGraph StateGraph** to enforce directed state machines, cycle limits, and durable checkpointing.

Specialized agents (Research, Data, Code, and Final Synthesis) operate under the dynamic supervision of a Supervisor agent, executing tools through a hardened execution registry equipped with call budgets, run-scoped tool failure isolation, and SSRF defenses.

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
        API -.-> OTel["OpenTelemetry Tracing<br/>(Spans & Context Propagation)"]
        ToolRegistry -.-> Guardrails["Tool Guardrails<br/>(Budget: 10 calls, Failures: 3, Payload limits)"]
        OrchestrationService -.-> CostTracker["Cost Tracker<br/>(Per-Model Token Accounting)"]
    end
```

---

## ⚡ Core Capabilities

- **State-Machine Orchestration**: Directed multi-agent collaboration with explicit loop limits (`MAX_ITERATIONS = 10`) and dynamic replanning via LangGraph.
- **Provider-Agnostic LLM Layer**: Unified interface supporting OpenRouter, Google Gemini, Groq, local models, and a deterministic offline Mock provider.
- **Model Context Protocol (MCP)**: Native integration with Anthropic's MCP specification, allowing dynamic tool discovery and execution via local or external MCP servers.
- **Hybrid Retrieval-Augmented Generation (RAG)**: Multi-format parsing (`.txt`, `.md`, `.pdf`), local ONNX dense embeddings (`all-MiniLM-L6-v2`), BM25 lexical search, and Reciprocal Rank Fusion (RRF $\alpha=0.6$).
- **Dual-Tier Memory**:
  - *Working Memory*: Thread execution history and scratchpads persisted to PostgreSQL via LangGraph's `AsyncPostgresSaver` (or in-memory `MemorySaver`).
  - *Long-Term Semantic Memory*: Persistent knowledge stored in Chroma (`agent_memory`), featuring entity extraction, cosine deduplication ($>0.85$), TTL sweeps, and composite ranking ($0.65\text{sim} + 0.25\text{imp} + 0.10\text{recency}$).
- **Human-in-the-Loop (HITL)**: Native LangGraph `interrupt()` gates triggered by sensitive actions or low supervisor confidence ($<0.65$), supporting asynchronous thread inspection and resumption.
- **Execution Replay & Time-Travel Debugging**: Historical checkpoint restoration, thread branching (`fork=True`), and mock tool injection for regression analysis.
- **Defense-in-Depth Security**: SSRF & DNS-rebinding protection on HTTP requests, AST-based arithmetic parsing without `eval()`, non-executing CodeAgent persona, payload size limits, and consecutive-failure tool disablement.
- **Distributed Observability**: OpenTelemetry spans across endpoints, graph steps, agent reasoning, and tool calls, complemented by token cost calculation and latency metrics.
- **Self-Healing Infrastructure**: Asynchronous connection pool re-initialization ensuring zero-downtime recovery when downstream databases start up out-of-order.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language & Runtime** | Python 3.12 (compatible with 3.11+) | Core platform programming language |
| **Web Framework** | FastAPI 0.110+, Uvicorn | High-performance ASGI REST API |
| **Orchestration** | LangGraph, LangChain Core | Cyclical state graphs, supervisor routing, checkpointing |
| **Data Validation** | Pydantic v2, Pydantic Settings | Typed request/response models and environment validation |
| **Tool Protocol** | Model Context Protocol (MCP) SDK v2 | Standardized tool discovery and execution |
| **Vector Database** | Chroma DB | Persistent embedding storage for RAG and semantic memory |
| **Embeddings** | ONNX Runtime (`all-MiniLM-L6-v2`) | Local, offline, zero-cost semantic embedding generation |
| **Relational Storage** | PostgreSQL 16 (`asyncpg`, `psycopg-pool`) | Persistent LangGraph working memory checkpointing |
| **Caching / Queues** | Redis 7 (configured, non-runtime-critical) | Auxiliary infrastructure cache |
| **Observability** | OpenTelemetry Python SDK | Distributed tracing, context propagation, span hierarchy |
| **Testing & Quality** | Pytest, Pytest-Asyncio, AnyIO, Ruff | 328 automated tests, static analysis, code formatting |
| **Containerization** | Docker, Docker Compose | Production non-root container and coordinated service stack |

---

## 📂 Repository Structure

```
multi-agent-orchestration-platform/
├── .dockerignore                 # Container build exclusions (.git, .venv, caches)
├── .env.example                  # Complete environment variable template
├── Dockerfile                    # Secure Python 3.12-slim image (non-root appuser)
├── docker-compose.yml            # Coordinated backend, PostgreSQL, and Redis stack
├── pyproject.toml                # Project packaging and Ruff configuration
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt          # Development, testing, and linting dependencies
├── README.md                     # Project overview and documentation index
├── benchmarks/
│   ├── README.md                 # Benchmark methodology and reproducibility notes
│   └── results/benchmark_report.json
├── demo/
│   └── data/                     # Canonical demo fixtures
├── docs/                         # Comprehensive engineering documentation
│   ├── README.md                 # Documentation navigation index
│   ├── architecture/             # System overview, runtime flow, component map, diagrams
│   ├── api/                      # Authoritative REST API specification (21 paths / 24 ops)
│   ├── development/              # Setup, configuration inventory, testing strategy
│   ├── operations/               # Deployment, health/readiness, troubleshooting
│   ├── security/                 # Threat model, SSRF protection, guardrails, non-goals
│   └── decisions/                # Architectural Decision Records (ADRs 001–005)
├── scripts/
│   ├── run_benchmarks.py         # Offline benchmark runner
│   ├── run_demo.py               # Canonical end-to-end demonstration
│   └── verify_deployment.py      # Deployment verification and smoke test
├── src/
│   └── app/
│       ├── main.py               # FastAPI application factory, lifespan, CORS, root routes
│       ├── api/                  # REST API routers and route controllers
│       ├── agents/               # Supervisor and specialist agent personas (Research, Data, Code, Final)
│       ├── orchestration/        # LangGraph StateGraph, typed state, node routing
│       ├── llm/                  # Multi-provider LLM abstraction (OpenRouter, Gemini, Groq, Mock)
│       ├── tools/                # Native tools (Calculator, SafeHTTP, RAG) and guarded registry
│       ├── mcp/                  # MCP Client, reference Server, and Tool Adapter
│       ├── rag/                  # Document parsing, recursive chunking, BM25 + ONNX hybrid retrieval
│       ├── memory/               # PostgreSQL working memory checkpointer and Chroma semantic store
│       ├── hitl/                 # Action approval policies, gating nodes, and resume handlers
│       ├── observability/        # OpenTelemetry tracing, system metrics, token pricing calculator
│       ├── evaluation/           # Offline evaluation framework and benchmark dataset runner
│       ├── replay/               # Checkpoint restoration, thread fork, and state override engine
│       ├── services/             # Cross-cutting business logic services
│       ├── core/                 # Typed configuration and structured logging
│       └── models/               # Domain Pydantic schemas and data contracts
└── tests/                        # 320 automated unit, integration, and E2E tests
```

---

## 🔌 API Surface Summary

The application exposes **21 unique URI paths** representing **24 HTTP operations**:

| Domain | Method | Endpoint Path | Description |
|---|---|---|---|
| **Health** | `GET` | `/health` | Lightweight process liveness probe |
| **Health** | `GET` | `/api/v1/health` | Comprehensive readiness probe with self-healing checks |
| **Agent** | `POST` | `/api/v1/agent/run` | Execute single tool-calling agent |
| **Orchestration** | `POST` | `/api/v1/orchestration/run` | Execute multi-agent LangGraph workflow |
| **Orchestration** | `POST` | `/api/v1/orchestration/replay` | Time-travel replay or fork from checkpoint |
| **HITL** | `GET` | `/api/v1/hitl/pending/{thread_id}` | Inspect paused thread requiring approval |
| **HITL** | `POST` | `/api/v1/hitl/resume/{thread_id}` | Submit human approval decision to resume |
| **Knowledge** | `POST` | `/api/v1/knowledge/documents` | Ingest raw text document into knowledge base |
| **Knowledge** | `GET` | `/api/v1/knowledge/documents` | List indexed documents |
| **Knowledge** | `POST` | `/api/v1/knowledge/documents/upload` | Multipart file upload (`.txt`, `.md`, `.pdf`) |
| **Knowledge** | `DELETE` | `/api/v1/knowledge/documents/{id}` | Delete document and vector chunks |
| **Knowledge** | `POST` | `/api/v1/knowledge/search` | Execute hybrid search (dense + BM25 + RRF) |
| **Knowledge** | `POST` | `/api/v1/knowledge/query` | RAG query returning synthesized answer |
| **Knowledge** | `GET` | `/api/v1/knowledge/stats` | Retrieve vector store statistics |
| **Memory** | `POST` | `/api/v1/memory` | Create new semantic long-term memory |
| **Memory** | `GET` | `/api/v1/memory` | List semantic memories by scope |
| **Memory** | `GET` | `/api/v1/memory/{id}` | Get memory item by ID |
| **Memory** | `DELETE` | `/api/v1/memory/{id}` | Delete memory item by ID |
| **Memory** | `POST` | `/api/v1/memory/search` | Semantic search with composite ranking |
| **Memory** | `POST` | `/api/v1/memory/consolidate` | Deduplicate vectors and purge expired TTL items |
| **Memory** | `GET` | `/api/v1/memory/stats` | Retrieve memory item metrics per scope |
| **MCP** | `GET` | `/api/v1/mcp/health` | Check status of local MCP server |
| **MCP** | `GET` | `/api/v1/mcp/tools` | List registered MCP tools and schemas |
| **LLM** | `GET` | `/api/v1/llm/providers` | Query active LLM provider and available models |

*Detailed request and response schemas are documented in [docs/api/overview.md](docs/api/overview.md).*

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 2. Installation
```bash
# Clone repository
git clone https://github.com/Karthik2509-git/multi-agent-orchestration-platform.git
cd multi-agent-orchestration-platform

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install runtime and development dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```
By default, `.env.example` is configured for **offline testing**:
```ini
LLM_PROVIDER=mock
MOCK_LLM_RESPONSE="Deterministic mock response for local testing"
DATABASE_URL=
CHROMA_PERSIST_DIR=./data/chroma
```
When `DATABASE_URL` is omitted, the platform uses an in-memory `MemorySaver` checkpointer.

### 4. Run the Development Server
```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the interactive documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🎬 Canonical Demo

The repository includes a single canonical demonstration script showcasing the platform's major capabilities as a coherent end-to-end engineering walkthrough, while keeping subsystem boundaries explicit rather than artificially coupling unrelated operations into a single execution. It exercises multi-agent coordination, guarded AST tool execution, hybrid RAG retrieval, semantic memory with scope isolation, Human-in-the-Loop approval gating, OpenTelemetry tracing and telemetry attribute redaction, retrieval evaluation metrics, and time-travel execution replay.

- **Offline & Zero-Cost**: 100% Offline, Deterministic Workflow Verification via the built-in `MockLLMProvider` and mock embeddings (no paid API keys, external networks, or Docker daemon required).
- **Single Command Execution**:
  ```bash
  python scripts/run_demo.py
  ```
- **Execution Time**: ~1-2 seconds across all 8 stages with clear pass/fail status.

*Detailed scenario architecture, stage explanations, and reproducibility guidelines are documented in [docs/development/canonical-demo.md](docs/development/canonical-demo.md).*

---

## 🧪 Testing & Verification

The test suite is built for **100% offline determinism** without external network calls or paid API keys.

```bash
# Run all 328 automated tests
pytest tests/

# Run code style and format checks
ruff check .
ruff format --check .
```

Current test suite status: **328 passed in ~50s** (0 failed, 0 skipped).

*Detailed test organization and methodology are documented in [docs/development/testing.md](docs/development/testing.md).*

---

## 📊 Benchmarks & Engineering Evidence

The platform provides a standalone benchmark runner executing 100% offline with deterministic evaluation/invariant suites and variable local runtime measurements, producing verifiable engineering evidence without external API calls or inflated marketing metrics:

- **Deterministic Retrieval Evaluation**: Measures hybrid search quality across curated benchmark fixtures (Recall@5: `0.6667`, Precision@5: `0.3817`, MRR: `0.6333`, Context-Assertion Overlap: `0.6673`).
- **Local Framework Processing Characterization**: In-process runtime latency profiles ($N=30$ iterations) measuring orchestration overhead (~47ms), hybrid RAG (~2.2ms), semantic memory search (~6.5ms), and AST calculator execution (~0.013ms). *Excludes real LLM/network inference; P99 represents an approximate upper-tail observation, not a production SLO.*
- **Run-Scoped Tool Failure Isolation**: Deterministically verifies execution limits (10 permitted calls, 11th blocked) and consecutive-failure tool disablement (3 errors disable tool; success resets streak).
- **Deterministic Cross-Scope Memory Isolation Verification**: Verifies zero cross-scope data leakage ($0.0\%$ leakage rate across tested isolation scenarios) under Chroma metadata segregation.
- **Versioned Token-Cost Projection**: Deterministic token expenditure projections using the versioned pricing registry (`MODEL_PRICING_REGISTRY`) under representative canonical workload assumptions.

```bash
# Execute complete benchmark suite and generate machine-readable JSON report
python scripts/run_benchmarks.py
```

*Detailed methodology, statistical interpretations, and claim boundaries are documented in [docs/development/benchmarks.md](docs/development/benchmarks.md). Machine-readable artifact: [benchmarks/results/benchmark_report.json](benchmarks/results/benchmark_report.json).*

---

## 🐳 Deployment & Operations

### Container Architecture
The platform is containerized using a multi-service `docker-compose.yml`:
- **PostgreSQL 16**: Backs the LangGraph checkpointer via persistent volume `postgres_data`.
- **Redis 7**: Auxiliary caching service bound to loopback `127.0.0.1:6379`.
- **Backend API**: Python 3.12-slim non-root container with `/app/data/chroma` mounted to named volume `chroma_data`.

### Deployment Verification Script
To verify an active deployment end-to-end:
```bash
python scripts/verify_deployment.py --base-url http://127.0.0.1:8000
```
This script exercises:
1. Liveness (`GET /health`)
2. Readiness (`GET /api/v1/health`)
3. RAG Ingestion & Query
4. Semantic Memory Storage & Search
5. Multi-Agent Orchestration Execution

*Detailed operational guidelines and honest verification boundaries are documented in [docs/operations/deployment.md](docs/operations/deployment.md).*

---

## ⚠️ Known Limitations & Boundaries

To ensure complete engineering transparency:

1. **Docker Runtime Verification**: Container configurations and Compose YAML syntax have been verified. However, live container startup and persistent named volume restart validation on the host machine were not executed during Phase 8 because the Docker Desktop daemon was offline.
2. **Telemetry Scope**: OpenTelemetry tracing and metrics collection currently operate in process-local mode unless an external OTLP collector endpoint is explicitly configured in `.env`.
3. **LLM Evaluation Judge**: The automated evaluation framework includes heuristic metrics and an optional LLM-as-a-judge component; the judge is invoked on-demand and is not run on live user traffic.
4. **Token Cost Modeling**: Token cost estimates are calculated from versioned pricing tables and rely on provider token reporting accuracy.
5. **Redis Role**: Redis is provisioned in Docker Compose and settings, but the backend does not currently depend on it for critical runtime operations.
6. **Authentication & Authorization**: The API endpoints currently operate without authentication or RBAC layers.
7. **No Web Frontend**: The platform is an API-first backend system and does not currently include a web UI.
8. **Replay Determinism**: When replaying threads with live external LLMs, completions are subject to model nondeterminism unless inputs and tool results are explicitly mocked.
9. **Security Scope**: Structural guardrails (SSRF filters, AST calculators, call budgets, payload caps) contain execution blasts but do not provide semantic prompt-injection classification.

---

## 🗺️ Future Roadmap

- **Production Observability Backends**: Production observability backends and dashboards, such as OTLP-compatible distributed tracing together with Prometheus/Grafana metrics visualization.
- **Operator Web Dashboard**: Lightweight web UI for visual inspection of LangGraph state trees, pending HITL approvals, and memory exploration.
- **Authentication & RBAC**: API key management and OAuth2 JWT authentication layer with granular endpoint permissions.
- **Cloud Infrastructure Templates**: Terraform configurations and Helm charts for deploying to Kubernetes (EKS / GKE).
- **Semantic Prompt Guardrails**: Integration of dedicated semantic classifiers for prompt injection and jailbreak detection.
- **Asynchronous Task Queuing**: Background worker processing using Celery or ARQ backed by the provisioned Redis infrastructure.
- **Multi-Tenant Memory Isolation**: Hard cryptographic and schema-level isolation for enterprise multi-tenancy.

---

## 📚 Complete Documentation Index

For detailed guides, please explore the `docs/` package:

- [System Architecture Overview](docs/architecture/overview.md)
- [Runtime Execution Flow & Alternate Branches](docs/architecture/runtime-flow.md)
- [Component & Subsystem Map](docs/architecture/component-map.md)
- [Architecture & Sequence Diagrams](docs/architecture/diagrams.md)
- [REST API Specification](docs/api/overview.md)
- [Local Development Setup](docs/development/setup.md)
- [Configuration Reference](docs/development/configuration.md)
- [Testing Strategy & Verification](docs/development/testing.md)
- [Platform Benchmarks & Engineering Evidence](docs/development/benchmarks.md)
- [Canonical Demo & Walkthrough Guide](docs/development/canonical-demo.md)
- [Deployment & Infrastructure Operations](docs/operations/deployment.md)
- [Health & Self-Healing Readiness Probes](docs/operations/health-and-readiness.md)
- [Operational Troubleshooting Guide](docs/operations/troubleshooting.md)
- [Platform Security Model](docs/security/security-model.md)
- [Architectural Decision Records (ADRs 001–005)](docs/decisions/)
