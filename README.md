# Multi-Agent AI Orchestration Platform

A production-grade, modular Multi-Agent AI Orchestration Platform designed to coordinate specialized AI agents, tool ecosystems, memory, and retrieval workflows using industry-standard engineering patterns.

---

## 🌟 Project Vision

The vision of this platform is to provide an enterprise-ready blueprint and runtime for multi-agent systems. It demonstrates how autonomous agents, supervised routing, external tool integrations, and human oversight can be orchestrated reliably at scale with rigorous observability, evaluation, and security guardrails.

### What the Platform Will Eventually Demonstrate
- **LLM Integration**: Unified multi-provider abstractions (OpenAI, Anthropic, Google Gemini, local models).
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
- **Guardrails & Security**: Prompt injection sanitization, PII filtering, SSRF protection, and policy enforcement.
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
│       │           ├── health.py # Health check endpoint (/api/v1/health)
│       │           └── agent.py  # Agent run endpoint (/api/v1/agent/run)
│       ├── models/              # Pydantic schemas and domain entities
│       │   └── schemas/
│       │       ├── health.py    # Health check data contracts
│       │       ├── agent.py     # Agent request and response contracts
│       │       └── llm.py       # LLM provider response and tool call contracts
│       ├── services/            # Reusable business logic services
│       │   ├── health.py        # Shared health status generator
│       │   └── agent_service.py # Agent task execution and tool registry factory
│       ├── agents/              # Agent layer
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
│       │       ├── openai.py    # OpenAI AsyncOpenAI provider
│       │       └── mock.py      # MockLLMProvider for offline deterministic tests
│       ├── orchestration/       # LangGraph state graphs and supervisors (Phase 3+)
│       ├── mcp/                 # Model Context Protocol clients (Phase 4+)
│       ├── rag/                 # Retrieval-Augmented Generation pipelines (Phase 5+)
│       ├── memory/              # Short-term and episodic memory systems (Phase 6+)
│       ├── db/                  # Database connections and repositories (Phase 5+)
│       ├── observability/       # Tracing, metrics, and monitoring (Phase 7+)
│       └── evaluation/          # Benchmark harnesses and eval suites (Phase 7+)
└── tests/                       # Pytest test suite (31 automated tests)
    ├── conftest.py              # Test client fixtures and environment overrides
    ├── test_health.py           # Health endpoint integration tests
    ├── test_calculator.py       # Calculator tool safety and arithmetic tests
    ├── test_http_tool.py        # HTTP tool SSRF, DNS-rebinding, and size tests
    ├── test_tool_registry.py    # ToolRegistry registration and execution tests
    ├── test_agent.py            # ToolCallingAgent loop and error recovery tests
    └── test_agent_api.py        # POST /api/v1/agent/run API endpoint tests
```

---

## 🚀 Current Implementation Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Foundation & Architecture** | ✅ Completed (Phase 1) | Modular directory structure, package hierarchy, and typing baseline |
| **FastAPI Core & Health** | ✅ Completed (Phase 1) | Application factory, lifespan events, CORS middleware, `/health` and `/api/v1/health` |
| **Configuration** | ✅ Completed (Phase 1) | Pydantic Settings (`BaseSettings`), environment variable validation, `.env.example` |
| **Containerization** | ✅ Completed (Phase 1) | Multi-stage Dockerfile (Python 3.12-slim, non-root user), Docker Compose with Postgres and Redis |
| **LLM Provider Abstraction** | ✅ Completed (Phase 2) | Clean `LLMProvider` interface with `OpenAILLMProvider` and `MockLLMProvider` |
| **Tool Registry & Safety** | ✅ Completed (Phase 2) | `ToolRegistry`, AST-based `calculator`, and SSRF-hardened `http_get` tool |
| **Single AI Agent** | ✅ Completed (Phase 2) | `ToolCallingAgent` with decision loop, validation, error recovery, and iteration ceiling |
| **Agent API** | ✅ Completed (Phase 2) | `POST /api/v1/agent/run` with structured responses, tool audit trails, and execution timing |
| **Test Suite** | ✅ Completed (Phase 2) | 31 unit and integration tests (zero live API key required for test suite) |
| **Orchestration / LangGraph** | ⏳ Pending (Phase 3) | Deferred to Phase 3 |
| **MCP Tool Ecosystem** | ⏳ Pending (Phase 4) | Deferred to Phase 4 |
| **RAG & Memory** | ⏳ Pending (Phases 5-6) | Deferred to respective phases |

---

## 🤖 Phase 2: LLM Integration & Tool-Calling Agent

Phase 2 introduces a single autonomous AI agent featuring real tool calling, strong security boundaries, and provider isolation:

```
User Task
   │
   ▼
[POST /api/v1/agent/run]
   │
   ▼
[Agent Service] ──► [ToolRegistry] (Calculator, Safe HTTP GET)
   │
   ▼
[ToolCallingAgent Loop]
   │
   ├─► 1. Send conversation history + tool schemas to LLMProvider
   │
   ├─► 2. LLM decides: Tool Call needed or Final Answer?
   │        │
   │        ├─► If Tool Call:
   │        │     - Validate tool name against registry
   │        │     - Validate arguments
   │        │     - Execute tool safely
   │        │     - Record duration & result
   │        │     - Append tool output to history
   │        │     - Re-prompt LLM (repeat up to max iterations)
   │        │
   │        └─► If Final Answer:
   │              - Record execution duration
   │              - Return structured AgentRunResponse
   ▼
FastAPI JSON Response
```

### 1. LLM Provider Architecture
All model interaction is mediated through the `LLMProvider` interface (`src/app/llm/base.py`). This prevents vendor lock-in and allows seamless switching between:
- **`OpenAILLMProvider`**: Live OpenAI integration using `AsyncOpenAI` with model name configurable via `OPENAI_MODEL` (`gpt-4o-mini` by default).
- **`MockLLMProvider`**: Deterministic offline provider for tests that supports canned responses or programmable mock handlers.

### 2. Built-In Tools & Security Hardening
- **Calculator Tool (`calculator`)**:
  - Uses Python's Abstract Syntax Tree (`ast`) module to evaluate expressions.
  - Strictly disallows `eval()` or `exec()`.
  - Blocks functions, imports, variables, and attribute access.
  - Enforces power exponent ceilings to prevent CPU/memory exhaustion denial-of-service attacks.
- **Safe HTTP GET Tool (`http_get`)**:
  - **SSRF Prevention**: Resolves hostnames before connecting and validates all resolved IP addresses against private (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), link-local (`169.254.0.0/16`), multicast, and reserved ranges.
  - **DNS-Rebinding Protection**: Validates every IP resolved from DNS to ensure no attacker-controlled host points to private internal infrastructure.
  - **Domain Allowlist**: Only permits domains explicitly listed in `ALLOWED_HTTP_DOMAINS`.
  - **Redirect Protection**: Automatic redirects are disabled (`follow_redirects=False`) to prevent redirect-based SSRF.
  - **Response Size Cap**: Reads up to `TOOL_HTTP_MAX_SIZE_BYTES` (default 100 KB) in chunks and truncates safely to prevent context window exhaustion.

### 3. Agent Execution Limits & Error Recovery
- **Iteration Ceiling**: Configurable via `AGENT_MAX_ITERATIONS` (default 5) to prevent infinite loops.
- **Tool Error Isolation**: Exceptions inside tools are captured and returned to the model as error messages, allowing the model to recover and explain issues gracefully.

---

## 🛠️ Tech Stack

- **Language**: Python 3.12 baseline (supports Python >=3.11)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) (0.110+)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (standard)
- **Validation & Settings**: [Pydantic v2](https://docs.pydantic.dev/latest/) & [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **LLM Integration**: [OpenAI Python SDK](https://github.com/openai/openai-python)
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
To test with live OpenAI models, set your `OPENAI_API_KEY` in `.env`:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
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

### Running an Agent Task via API
```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Calculate (1000 / 8) + 42"}'
```

Example response:
```json
{
  "task": "Calculate (1000 / 8) + 42",
  "answer": "The result of (1000 / 8) + 42 is 167.",
  "tool_calls": [
    {
      "tool_name": "calculator",
      "arguments": { "expression": "(1000 / 8) + 42" },
      "result": { "expression": "(1000 / 8) + 42", "result": 167.0 },
      "success": true,
      "error": null,
      "duration_ms": 0.42
    }
  ],
  "model": "gpt-4o-mini",
  "status": "completed",
  "execution_time_seconds": 0.8532
}
```

---

## 🧪 Running Tests

Execute the automated test suite with `pytest`:
```bash
# Run all 31 tests
pytest -v

# Run linting check
ruff check .
```

All unit and integration tests use `MockLLMProvider` and mock HTTP transports, allowing the entire suite to run with zero external API dependencies or API keys.

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
- **Phase 3 — Multi-Agent Orchestration with LangGraph**
  - StateGraph design, supervisor routing, specialist worker agents, and deterministic loop control.
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
