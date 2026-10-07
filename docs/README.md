# Documentation Index

Welcome to the technical documentation for the **Multi-Agent AI Orchestration Platform**. This repository provides an enterprise-grade reference architecture for multi-agent systems, combining directed graph orchestration, tool ecosystems, hybrid retrieval, dual-tier memory, and operational governance.

---

## Documentation Directory

### 🏛️ Architecture
- [System Architecture Overview](architecture/overview.md) — High-level system structure, layer breakdown, and Mermaid topology diagram.
- [Runtime Execution Flow](architecture/runtime-flow.md) — Step-by-step lifecycle of orchestration requests and 10 alternate branches.
- [Component & Subsystem Map](architecture/component-map.md) — Comprehensive breakdown of code directories and subsystem responsibilities.
- [Architecture Diagrams](architecture/diagrams.md) — Collection of Mermaid visual diagrams covering orchestration, RAG, memory, HITL, guardrails, and replay.

### 🔌 API Reference
- [REST API Specification](api/overview.md) — Comprehensive reference for all 21 URI paths and 24 HTTP operations.

### 💻 Development & Engineering
- [Canonical Demonstration & Reproducibility Guide](development/canonical-demo.md) — Single deterministic end-to-end scenario exercising all 8 core platform capabilities.
- [Local Development Setup](development/setup.md) — Prerequisites, virtual environment setup, mock provider usage, and running locally.
- [Configuration Reference](development/configuration.md) — Complete inventory of environment variables, settings, and validators.
- [Testing Strategy & Verification](development/testing.md) — Automated test inventory, offline determinism, and linting guidelines.
- [Benchmark Methodology & Engineering Evidence](development/benchmarks.md) — Deterministic evaluation, invariant verification, local runtime characterization, and cost projections.

### 🚀 Operations & Deployment
- [Deployment & Infrastructure](operations/deployment.md) — Docker containerization, Docker Compose architecture, volumes, and deployment verification script.
- [Health & Self-Healing Readiness](operations/health-and-readiness.md) — Process liveness vs. deep readiness, and asynchronous PostgreSQL pool self-healing.
- [Operational Troubleshooting](operations/troubleshooting.md) — Common failure modes, root cause analyses, and operational remedies.

### 🛡️ Security & Governance
- [Security Model & Defense-in-Depth](security/security-model.md) — SSRF protection, AST arithmetic parser, CodeAgent zero-execution guarantee, tool execution budgets, and security boundaries.

### 📜 Architectural Decision Records (ADRs)
- [ADR 001: LangGraph for Multi-Agent Orchestration](decisions/001-langgraph-orchestration.md)
- [ADR 002: Model Context Protocol (MCP) as Extensible Tool Transport](decisions/002-mcp-tool-architecture.md)
- [ADR 003: Hybrid Retrieval-Augmented Generation (RAG) Architecture](decisions/003-hybrid-rag.md)
- [ADR 004: Dual-Tier Memory and Human-in-the-Loop Architecture](decisions/004-memory-and-hitl.md)
- [ADR 005: Observability, Guardrails, Evaluation and Replay Architecture](decisions/005-observability-and-replay.md)
