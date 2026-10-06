# Testing Strategy & Verification

The platform maintains an automated test suite designed for **complete offline determinism, high execution speed, and comprehensive boundary verification**.

---

## 1. Test Suite Summary

- **Total Automated Tests**: **320 passed** (0 failures, 0 skipped).
- **Execution Time**: ~50 seconds for the entire test suite on standard development hardware.
- **External Dependency Requirement**: **Zero**. All tests run completely offline without external network connectivity, live API keys, paid credits, or container daemons.
- **Code Coverage Note**: While tests exercise all core functional modules, code coverage percentages are not published to prevent claiming unmeasured metrics.

---

## 2. Test Architecture & Pytest Organization

The tests are organized in `tests/` across specialized domain suites:

```
tests/
├── test_agent.py                         # Single ToolCallingAgent execution & prompt flow
├── test_agent_api.py                     # Agent REST endpoint contract tests
├── test_agent_multiprovider.py           # Provider selection & fallback validation
├── test_calculator.py                    # AST calculator parser, limits, and security tests
├── test_deployment_reliability.py        # End-to-end deployment verification script harness
├── test_e2e_integration.py               # Complete cross-subsystem orchestration tests
├── test_evaluation_framework.py          # Benchmark dataset and evaluation runner tests
├── test_execution_replay.py              # Time-travel checkpoint replay and fork tests
├── test_gemini_provider.py               # Google GenAI provider adapter tests
├── test_groq_provider.py                 # Groq LPU provider adapter tests
├── test_health.py                        # Health & readiness endpoint verification
├── test_hitl_api.py                      # HITL inspection & resume API tests
├── test_hitl_policies.py                 # Action gating & confidence threshold tests
├── test_hitl_workflow.py                 # LangGraph interrupt() lifecycle tests
├── test_http_tool.py                     # SSRF, DNS-rebinding, and payload limit tests
├── test_knowledge_api.py                 # RAG document upload, query, and search endpoints
├── test_knowledge_search_tool.py         # KnowledgeSearchTool agent integration
├── test_llm_factory_and_status.py        # LLM provider factory and status inspection
├── test_mcp_adapter.py                   # MCP tool to BaseTool conversion tests
├── test_mcp_api.py                       # MCP health and tools endpoints
├── test_mcp_client.py                    # MCP Client protocol tests
├── test_mcp_models.py                   # MCP data schema and serialization tests
├── test_mcp_registry.py                  # MCP server registry and lifecycle tests
├── test_mcp_server.py                    # In-process reference MCP server tests
├── test_mcp_service.py                   # MCP service layer tests
├── test_mcp_tool_integration.py          # Multi-agent MCP tool execution tests
├── test_memory_api.py                    # Long-term memory CRUD and stats endpoints
├── test_memory_consolidation.py          # Memory deduplication & TTL sweep tests
├── test_memory_extraction.py             # Entity and constraint extraction heuristics
├── test_memory_models.py                 # Memory schema and validation tests
├── test_memory_retrieval.py              # Composite ranking (similarity + importance + recency)
├── test_memory_store.py                  # Chroma semantic memory store integration
├── test_observability_cost.py            # Token pricing attribution and cost tracking
├── test_observability_metrics.py         # System metrics collector tests
├── test_observability_milestone1.py      # OpenTelemetry span generation tests
├── test_openrouter_provider.py           # OpenRouter provider adapter tests
├── test_orchestration_api.py             # Orchestration run endpoint contract tests
├── test_orchestration_graph.py           # LangGraph StateGraph transition tests
├── test_orchestration_phase6_integration.py # Orchestration + Memory + HITL integration
├── test_orchestration_state.py           # State schema and history serialization
├── test_production_config.py             # Production password and settings validation
├── test_rag_chunking.py                  # Recursive character chunker boundary tests
├── test_rag_embeddings.py               # Local ONNX and mock embedding provider tests
├── test_rag_hybrid_retriever.py          # Reciprocal Rank Fusion & BM25 ranking tests
├── test_rag_ingestion.py                 # Document parser (.txt, .md, .pdf) tests
├── test_rag_models.py                    # RAG document and search schema tests
├── test_rag_service.py                   # Ingestion and query service orchestration
├── test_rag_tool_agent_integration.py     # Specialist agent RAG querying tests
├── test_rag_vector_store.py              # ChromaVectorStore persistence tests
├── test_specialized_agents.py            # Specialist persona logic (Research, Data, Code, Final)
├── test_supervisor.py                    # Supervisor intent routing and loop termination
├── test_tool_execution_context.py        # Execution tracking and context isolation
├── test_tool_guardrails.py               # Call budgets, consecutive-failure tool disablement, and payload limits
├── test_tool_registry.py                 # Tool registration and error isolation tests
└── test_working_memory.py                # LangGraph checkpointer concurrency & fallback tests
```

---

## 3. Running the Test Suite

### Run All 320 Tests
```bash
pytest
```

### Run Verbose with Test Names
```bash
pytest -v
```

### Run Specific Test Domain
```bash
# Test multi-agent orchestration only
pytest tests/test_orchestration_*.py

# Test security guardrails & SSRF protection
pytest tests/test_http_tool.py tests/test_calculator.py tests/test_tool_guardrails.py

# Test deployment reliability & smoke harness
pytest tests/test_deployment_reliability.py
```

---

## 4. Code Formatting & Linting

The repository strictly adheres to **Ruff** for linting and formatting standards.

```bash
# Run static lint check
ruff check .

# Run code format verification
ruff format --check .

# Automatically apply safe fixes and formatting
ruff check --fix .
ruff format .
```
