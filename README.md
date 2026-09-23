# Multi-Agent AI Orchestration Platform

A production-grade, modular Multi-Agent AI Orchestration Platform designed to coordinate specialized AI agents, tool ecosystems, memory, and retrieval workflows using industry-standard engineering patterns.

---

## 🌟 Project Vision

The vision of this platform is to provide an enterprise-ready blueprint and runtime for multi-agent systems. It demonstrates how autonomous agents, supervised routing, external tool integrations, and human oversight can be orchestrated reliably at scale with rigorous observability, evaluation, and security guardrails.

### What the Platform Will Eventually Demonstrate
- **LLM Integration**: Unified multi-provider abstractions (OpenRouter, Google Gemini, Groq, local models).
- **Multi-Agent Architecture**: Hierarchical supervisor and specialist agents cooperating on complex objectives.
- **LangGraph Orchestration**: State-machine-driven agent control loops, branching, cycles, and deterministic graph flows.
- **Supervisor & Specialist Agents**: Domain-focused workers (research, code generation, analysis, verification).
- **Model Context Protocol (MCP)**: Native MCP client integration for standardized external tooling and context servers.
- **Tool / Function Calling**: Typed schema generation, dynamic tool registry, input validation, and execution guards.
- **REST API Integration**: FastAPI endpoints exposing asynchronous agent runs, job status polling, and streaming responses.
- **RAG (Retrieval-Augmented Generation)**: Ingestion pipelines, hybrid vector search, chunking strategies, and re-ranking.
- **Vector Databases**: Scalable document indexing and similarity search.
- **Short-Term & Long-Term Memory**: Conversation checkpointing, episodic storage, and semantic user memory retrieval.
- **Human-in-the-Loop Workflows**: Interrupt hooks, approval gates, and state editing for sensitive actions.
- **Confidence-Based Routing**: Dynamic routing based on agent self-evaluation and uncertainty heuristics.
- **Retries & Error Handling**: Exponential backoff, fallback models, and graceful degradation.
- **Async / Background Task Execution**: Decoupled task queue processing for long-running workflows.
- **Observability & Tracing**: Distributed tracing with OpenTelemetry, span tracking for LLM calls, and metrics dashboards.
- **Evaluation**: Agent benchmark suites, output quality metrics, and automated regression testing.
- **Guardrails & Security**: Prompt injection sanitization, PII filtering, SSRF/DNS-rebinding protection, and policy enforcement.
- **Infrastructure**: FastAPI, PostgreSQL, Redis, Docker, Automated Testing, and CI/CD pipelines.

---

## 🏛️ High-Level Architecture

The platform follows clean architecture principles with strict separation of concerns, dependency injection, and modular encapsulation:

```
multi-agent-orchestration-platform/
├── .env.example                 # Canonical environment configuration template
├── .gitignore                   # Production ignore patterns (secrets, virtualenvs, cache)
├── Dockerfile                   # Production-grade Python 3.12-slim container
├── docker-compose.yml           # Backend, PostgreSQL, and Redis infrastructure
├── pyproject.toml               # Modern Python packaging and tool configuration
├── requirements.txt             # Pinned production dependencies
├── requirements-dev.txt         # Development, testing, and linting dependencies
├── README.md                    # Platform documentation and roadmap
├── src/
│   └── app/
│       ├── __init__.py          # Package initialization (__version__ = "0.1.0")
│       ├── main.py              # Application factory, lifespan, CORS, and root routes
│       ├── core/                # Typed configuration & centralized logging
│       │   ├── config.py        # Pydantic Settings management
│       │   └── logging.py       # Structured logging setup
│       ├── api/                 # REST API layer with versioning
│       │   └── v1/
│       │       ├── api.py       # Aggregation router for v1
│       │       └── endpoints/
│       │           ├── health.py        # Health check endpoint (/api/v1/health)
│       │           ├── agent.py         # Agent run endpoint (/api/v1/agent/run)
│       │           ├── llm.py           # Provider status endpoint (/api/v1/llm/providers)
│       │           └── orchestration.py # Multi-agent orchestration (/api/v1/orchestration/run)
│       ├── models/              # Pydantic schemas and domain entities
│       │   └── schemas/
│       │       ├── health.py        # Health check data contracts
│       │       ├── agent.py         # Single agent request and response contracts
│       │       ├── llm.py           # LLM provider response and tool call contracts
│       │       ├── provider.py      # Provider configuration status schema
│       │       └── orchestration.py # Multi-agent orchestration run contracts
│       ├── services/            # Reusable business logic services
│       │   ├── health.py                # Shared health status generator
│       │   ├── agent_service.py         # Agent task execution and tool registry factory
│       │   └── orchestration_service.py # LangGraph workflow coordinator
│       ├── agents/              # Agent layer
│       │   ├── base.py               # BaseSpecializedAgent and AgentResult
│       │   ├── supervisor.py         # SupervisorAgent with strict allowlist routing
│       │   ├── research_agent.py     # ResearchAgent with safe URL retrieval
│       │   ├── data_agent.py         # DataAgent with CalculatorTool integration
│       │   ├── code_agent.py         # CodeAgent with zero-execution guarantee
│       │   ├── final_agent.py        # FinalAgent aggregator and synthesizer
│       │   └── tool_calling_agent.py # Single AI agent with tool loop
│       ├── tools/               # Agent tool implementations & registry
│       │   ├── base.py          # BaseTool and ToolResult abstractions
│       │   ├── registry.py      # Tool discovery and execution registry
│       │   ├── calculator.py    # Safe AST-based arithmetic calculator
│       │   └── http_tool.py     # Safe HTTP GET with SSRF and DNS-rebinding protection
│       ├── llm/                 # Unified LLM provider adapters
│       │   ├── base.py          # LLMProvider abstract interface
│       │   ├── factory.py       # LLM provider dependency injection factory
│       │   └── providers/
│       │       ├── openrouter.py # OpenRouter provider (OpenAI-compatible async client)
│       │       ├── gemini.py     # Google Gemini provider (official google-genai SDK)
│       │       ├── groq.py       # Groq provider (ultra-fast LPU inference)
│       │       ├── mock.py       # MockLLMProvider for offline deterministic tests
│       │       └── openai.py     # Optional legacy OpenAI provider
│       ├── orchestration/       # LangGraph multi-agent workflows
│       │   ├── state.py         # Shared typed state (OrchestrationState)
│       │   └── graph.py         # StateGraph assembly, worker routing & loop ceiling
│       ├── mcp/                 # Model Context Protocol (MCP) subsystem
│       │   ├── models.py        # Normalized MCP tool definition & server status contracts
│       │   ├── client.py        # MCPClient wrapper delegating to official v2 Client
│       │   ├── adapters.py      # MCPToolAdapter making MCP tools compatible with BaseTool
│       │   ├── registry.py      # MCPToolRegistry managing discovery, allowlisting, and injection
│       │   └── servers/         # In-process MCPServers
│       │       └── local_tools.py # Local MCPServer with safe AST calculator & text_stats
│       ├── rag/                 # Retrieval-Augmented Generation pipelines (Phase 5+)
│       ├── memory/              # Short-term and episodic memory systems (Phase 6+)
│       ├── db/                  # Database connections and repositories (Phase 5+)
│       ├── observability/       # Tracing, metrics, and monitoring (Phase 7+)
│       └── evaluation/          # Benchmark harnesses and eval suites (Phase 7+)
└── tests/                       # Pytest test suite (96 automated tests)
    ├── conftest.py                    # Test client fixtures and environment overrides
    ├── test_health.py                 # Health endpoint integration tests
    ├── test_calculator.py             # Calculator tool safety and arithmetic tests
    ├── test_http_tool.py              # HTTP tool SSRF, DNS-rebinding, and size tests
    ├── test_tool_registry.py          # ToolRegistry registration and execution tests
    ├── test_agent.py                  # ToolCallingAgent loop and error recovery tests
    ├── test_agent_api.py              # POST /api/v1/agent/run API endpoint tests
    ├── test_openrouter_provider.py    # OpenRouter provider normalization & error tests
    ├── test_groq_provider.py          # Groq provider normalization & error tests
    ├── test_gemini_provider.py        # Gemini SDK normalization & tool mapping tests
    ├── test_llm_factory_and_status.py # Factory selection and provider status tests
    ├── test_agent_multiprovider.py    # Cross-provider agent tool execution tests
    ├── test_orchestration_state.py    # OrchestrationState typed structure tests
    ├── test_supervisor.py             # Strict allowlist parsing and fallback tests
    ├── test_specialized_agents.py     # Research, Data, Code, and Final agent tests
    ├── test_orchestration_graph.py    # LangGraph flows, loop ceiling, and state updates
    ├── test_orchestration_api.py      # POST /api/v1/orchestration/run endpoint tests
    ├── test_mcp_models.py             # MCPToolDefinition and MCPServerStatus model tests
    ├── test_mcp_server.py             # In-process MCPServer and safe arithmetic tests
    ├── test_mcp_client.py             # MCPClient v2 Client connection and call tests
    ├── test_mcp_adapter.py            # MCPToolAdapter BaseTool compatibility tests
    ├── test_mcp_registry.py           # MCPToolRegistry allowlisting and collision tests
    ├── test_mcp_service.py            # MCPService lifecycle and status tests
    ├── test_mcp_api.py                # GET /api/v1/mcp/health and /tools endpoint tests
    └── test_mcp_tool_integration.py   # ToolCallingAgent executing mcp.local.calculator
```

---

## 🚀 Current Implementation Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Foundation & Architecture** | ✅ Completed (Phase 1) | Modular directory structure, package hierarchy, and typing baseline |
| **FastAPI Core & Health** | ✅ Completed (Phase 1) | Application factory, lifespan events, CORS middleware, `/health` and `/api/v1/health` |
| **Configuration** | ✅ Completed (Phase 1) | Pydantic Settings (`BaseSettings`), environment variable validation, `.env.example` |
| **Containerization** | ✅ Completed (Phase 1) | Multi-stage Dockerfile (Python 3.12-slim, non-root user), Docker Compose with Postgres and Redis |
| **Tool Registry & Safety** | ✅ Completed (Phase 2) | `ToolRegistry`, AST-based `calculator`, and SSRF/DNS-rebinding hardened `http_get` tool |
| **Single AI Agent** | ✅ Completed (Phase 2) | `ToolCallingAgent` with decision loop, validation, error recovery, and iteration ceiling |
| **Agent API** | ✅ Completed (Phase 2) | `POST /api/v1/agent/run` with structured responses, tool audit trails, and execution timing |
| **Multi-Provider LLM Infrastructure** | ✅ Completed (Phase 2.5) | Provider-agnostic architecture: OpenRouter, Google Gemini, Groq, Mock, and OpenAI |
| **Provider Status Endpoint** | ✅ Completed (Phase 2.5) | `GET /api/v1/llm/providers` exposing active provider and configuration readiness |
| **Multi-Agent Orchestration (LangGraph)** | ✅ Completed (Phase 3) | StateGraph workflow, SupervisorAgent, specialized workers, loop ceiling, and orchestration API |
| **MCP Tool Ecosystem** | ✅ Completed (Phase 4) | Official MCP SDK v2 Client & MCPServer, `MCPToolAdapter`, allowlisting, namespacing & REST APIs |
| **Test Suite** | ✅ Completed (Phase 4) | 96 unit and integration tests (zero paid API keys required for testing) |
| **RAG & Knowledge System** | ⏳ Pending (Phase 5) | Deferred to Phase 5 |
| **Memory & HITL** | ⏳ Pending (Phase 6) | Deferred to Phase 6 |

---

## 🔌 Phase 2.5: Multi-Provider LLM Infrastructure

Phase 2.5 establishes a provider-agnostic infrastructure. The `ToolCallingAgent` has zero provider-specific logic and communicates exclusively through the abstract `LLMProvider` interface.

```
                    ToolCallingAgent
                           │
                      LLMProvider
                           │
          ┌────────────────┼────────────────┬────────────────┐
          ▼                ▼                ▼                ▼
     OpenRouter          Gemini            Groq             Mock
  (openrouter/free) (gemini-2.5-flash) (llama-3.3-70b)   (Offline Tests)
```

### Supported Providers

1. **OpenRouter (`openrouter`)** — *Default*:
   - Uses OpenAI-compatible async client with base URL `https://openrouter.ai/api/v1`.
   - Includes OpenRouter recommended headers (`HTTP-Referer`, `X-Title`).
   - Configurable model via `OPENROUTER_MODEL` (e.g. `openrouter/free` or `meta-llama/llama-3.3-70b-instruct:free`).
2. **Google Gemini (`gemini`)**:
   - Uses the official `google-genai` SDK (`genai.Client`).
   - Maps user messages to Gemini `user` content, assistant text/calls to `model` content, and tool results to `function_response` content.
   - Translates tool definitions into Gemini `FunctionDeclaration` objects.
   - Configurable model via `GEMINI_MODEL` (e.g. `gemini-2.5-flash`).
3. **Groq (`groq`)**:
   - Uses OpenAI-compatible async client with base URL `https://api.groq.com/openai/v1`.
   - Ultra-low latency LPU inference on free-tier open models.
   - Configurable model via `GROQ_MODEL` (e.g. `llama-3.3-70b-versatile`).
4. **Mock (`mock`)**:
   - Requires zero API keys or network calls; used for deterministic testing of tool cycles and error states.
5. **OpenAI (`openai`)**:
   - Maintained as an optional legacy provider; never required for running the platform.

### Zero Paid Requirement & Free-Tier Development
The platform is designed to be fully runnable with zero paid API providers. All model identifiers are environment-variable driven:
- `LLM_PROVIDER`: Sets the active provider (`openrouter`, `gemini`, `groq`, `mock`, `openai`).
- `OPENROUTER_MODEL`: Model identifier on OpenRouter.
- `GEMINI_MODEL`: Model identifier on Google Gemini.
- `GROQ_MODEL`: Model identifier on Groq.

Missing credentials on *inactive* providers will never cause import or runtime failures. If `LLM_PROVIDER=openrouter` is active, empty Gemini and Groq keys are completely ignored.

---

## 📡 Provider Status API Endpoint

Inspect provider configuration and active status:
```bash
curl http://127.0.0.1:8000/api/v1/llm/providers
```

Example response (zero credentials exposed):
```json
{
  "active_provider": "openrouter",
  "providers": {
    "openrouter": {
      "configured": true,
      "model": "openrouter/free"
    },
    "gemini": {
      "configured": false,
      "model": "gemini-2.5-flash"
    },
    "groq": {
      "configured": false,
      "model": "llama-3.3-70b-versatile"
    },
    "mock": {
      "configured": true,
      "model": "mock-model"
    },
    "openai": {
      "configured": false,
      "model": "gpt-4o-mini"
    }
  }
}
```

---

## 🧠 Phase 3: Multi-Agent Orchestration with LangGraph

Phase 3 transitions the platform from a single tool-calling agent to a modular, state-driven multi-agent orchestration graph built on **LangGraph**.

### Graph Topology & State Machine

```
               START
                 │
                 ▼
         ┌───────────────┐
         │  Supervisor   │◄──────────────────────────┐
         └───────┬───────┘                           │
                 │                                   │
      ┌──────────┼──────────┐                        │
      ▼          ▼          ▼                        │
┌──────────┐┌──────────┐┌──────────┐                 │
│ Research ││   Data   ││   Code   │                 │
│  Agent   ││  Agent   ││  Agent   │                 │
└────┬─────┘└────┬─────┘└────┬─────┘                 │
     │           │           │                       │
     └───────────┴───────────┴───────────────────────┘
                 │
          (next_agent: final / loop limit)
                 │
                 ▼
          ┌─────────────┐
          │    Final    │
          │ Synthesizer │
          └──────┬──────┘
                 │
                 ▼
                END
```

### Orchestration Components

- **Typed Orchestration State (`OrchestrationState`)**:
  - Encapsulates `task`, `messages`, `next_agent`, `agent_results` dict, `agents_used` list, `step_count`, `final_answer`, `status`, and non-sensitive `metadata`.
- **Defensive Supervisor Routing (`SupervisorAgent`)**:
  - Evaluates user objectives and accumulated specialist findings.
  - Strict allowlist parser enforcing only valid targets: `research`, `data`, `code`, `final`.
  - Malformed model responses, unexpected strings (e.g., `browser`), or parse errors automatically and safely default to `final`.
- **Specialized Worker Agents**:
  - **`ResearchAgent`**: Analyzes research tasks and optionally inspects explicitly permitted URLs using `SafeHTTPGetTool`, synthesizing findings with the active LLM. (Web search is deferred to Phase 4 MCP).
  - **`DataAgent`**: Handles quantitative and numerical analysis, safely reusing `CalculatorTool` for arithmetic verification.
  - **`CodeAgent`**: Specializes in code architecture, syntax review, and implementation generation. Strictly guarantees zero code execution and zero shell interaction.
  - **`FinalAgent`**: Aggregates all specialist findings into a unified, user-facing synthesized response.
- **Deterministic Loop Ceiling**:
  - Enforces `MAX_ORCHESTRATION_STEPS = 8`. If the supervisor exceeds 8 routing transitions, the workflow forces termination to `final` and records `terminated_due_to_limit: True` in metadata.
- **Provider-Agnostic Core**:
  - All supervisor and specialist nodes depend strictly on the abstract `LLMProvider` interface.

---

## 🤖 Built-In Tools & Security Hardening

- **Calculator Tool (`calculator`)**:
  - Uses Python's Abstract Syntax Tree (`ast`) module to safely parse and evaluate arithmetic expressions.
  - Strictly disallows `eval()` or `exec()`.
  - Blocks functions, imports, variables, and attribute access.
  - Enforces power exponent ceilings to prevent CPU/memory exhaustion denial-of-service attacks.
- **Safe HTTP GET Tool (`http_get`)**:
  - **SSRF Prevention**: Resolves hostnames before connecting and validates all resolved IP addresses against private (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), link-local (`169.254.0.0/16`), multicast, and reserved ranges.
  - **DNS-Rebinding Protection**: Validates every IP resolved from DNS to ensure no attacker-controlled host points to private internal infrastructure.
  - **Domain Allowlist**: Only permits domains explicitly listed in `ALLOWED_HTTP_DOMAINS` (`httpbin.org`, `api.github.com`).
  - **Redirect Protection**: Automatic redirects are disabled (`follow_redirects=False`) to prevent redirect-based SSRF.
  - **Response Size Cap**: Streams response chunks up to `TOOL_HTTP_MAX_SIZE_BYTES` (default 100 KB) and truncates safely.

---

## 🛠️ Tech Stack

- **Language**: Python 3.12 baseline (supports Python >=3.11)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (0.110+)
- **Multi-Agent Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph & typed state workflows)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (standard)
- **Validation & Settings**: [Pydantic v2](https://docs.pydantic.dev/latest/) & [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **LLM Integrations**: [OpenAI Python SDK](https://github.com/openai/openai-python) (for OpenRouter & Groq), [Google GenAI SDK](https://github.com/googleapis/python-genai) (for Gemini)
- **HTTP Client**: [HTTPX](https://www.python-httpx.org/)
- **Testing**: [Pytest](https://docs.pytest.org/), [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)
- **Code Quality**: [Ruff](https://astral.sh/ruff)
- **Infrastructure**: [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)
- **Planned Infrastructure**: PostgreSQL 16, Redis 7

---

## ⚙️ Local Setup Instructions

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13 installed
- Git
- Docker and Docker Compose (optional for local running, required for containerized deployment)

### 2. Clone the Repository
```bash
git clone https://github.com/Karthik2509-git/multi-agent-orchestration-platform.git
cd multi-agent-orchestration-platform
```

### 3. Create and Activate a Virtual Environment
```bash
# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
# Install runtime and development dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Choose your active provider in `.env`:
```bash
# Select active provider
LLM_PROVIDER=openrouter

# OpenRouter (Free-tier models available)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/free

# Or Google Gemini
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-2.5-flash

# Or Groq
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_key_here
# GROQ_MODEL=llama-3.3-70b-versatile
```

---

## 🏃 Running the Application

### Local Development Server
Start the FastAPI application with auto-reload:
```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Running Single-Agent Task via API
```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Calculate (1000 / 8) + 42"}'
```

### Running Multi-Agent Orchestration via API
```bash
curl -X POST http://127.0.0.1:8000/api/v1/orchestration/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze Apple revenue expansion and calculate the CAGR from 2021 to 2023"}'
```

Example multi-agent response:
```json
{
  "task": "Analyze Apple revenue expansion and calculate the CAGR from 2021 to 2023",
  "answer": "Apple grew revenues from $365.8B in 2021 to $383.3B in 2023. The calculated CAGR across this period is 2.37%...",
  "agents_used": ["research", "data", "final"],
  "status": "completed",
  "execution_time_seconds": 1.742,
  "metadata": null
}
```

---

## 🧪 Running Tests

Execute the automated test suite with `pytest`:
```bash
# Run all 76 tests
pytest -v

# Run linting check and code formatting verification
ruff check .
ruff format --check .
```

All 76 unit and integration tests run offline with zero external API calls or paid credentials.

---

## 🐳 Docker Instructions

### Build the Docker Image
```bash
docker build -t multi-agent-orchestration-platform:latest .
```

### Run Full Infrastructure with Docker Compose
```bash
docker compose up -d --build
```

Check service status and health:
```bash
docker compose ps
```

---

## 🗺️ Future Development Phases

The platform is engineered iteratively phase-by-phase. Future development will follow this roadmap:

- **Phase 1 — Foundation & Project Architecture** *(Completed)*
  - Modular project structure, typing, configuration, health endpoints, containerization, and test harnesses.
- **Phase 2 — LLM Integration & Tool-Calling Agent** *(Completed)*
  - Unified LLM provider abstraction, tool registry, safe AST calculator, SSRF-hardened HTTP tool, ToolCallingAgent, and REST endpoint.
- **Phase 2.5 — Multi-Provider LLM Infrastructure** *(Completed)*
  - Provider-agnostic architecture supporting OpenRouter, Google Gemini, Groq, and Mock with free-tier model support and provider status API.
- **Phase 3 — Multi-Agent Orchestration with LangGraph** *(Completed)*
  - LangGraph StateGraph design, SupervisorAgent with strict allowlist routing, specialist worker agents (Research, Data, Code, Final), deterministic loop ceiling, and REST orchestration API.
- **Phase 4 — MCP Tool Ecosystem**
  - Model Context Protocol (MCP) clients, adapters, and standardized external resource integration.
- **Phase 5 — RAG & Knowledge System**
  - Document parsing, vector database indexing, hybrid semantic search, and retrieval pipelines.
- **Phase 6 — Memory & Human-in-the-Loop**
  - State checkpointing, long-term episodic memory, interruption breakpoints, and human review gates.
- **Phase 7 — Observability, Guardrails & Evaluation**
  - OpenTelemetry distributed tracing, LLM cost/latency tracking, security guardrails, and automated evaluation suites.
- **Phase 8 — Production Deployment & Portfolio Polish**
  - Production hardening, cloud deployment automation, CI/CD pipelines, and end-to-end multi-agent showcase.
