# Canonical Demonstration & Reproducibility Guide

This guide documents the **canonical demonstration** of the Multi-Agent AI Orchestration Platform, providing a single reproducible end-to-end scenario that exercises the platform's multi-agent coordination, tool execution, hybrid RAG, memory, safety gating, observability, evaluation, and time-travel replay.

---

## 1. Purpose & Target Audience

- **Recruiters & Engineering Leaders**: Observe the system's architecture, state transitions, and safety boundaries in a fast (~1-2 second) local execution.
- **Technical Interviewers**: Inspect real LangGraph agent loops, guarded AST tool invocations, hybrid retrieval, and thread checkpoint forks without mock hand-waving.
- **Developers & Contributors**: Verify local installation integrity and reproduce system capabilities with zero external API credentials or cloud dependencies.
- **Demo Recording**: Provides a clean, structured console output designed for screen capture and portfolio presentation.

---

## 2. Canonical Scenario: Project Titan Architecture Audit & Sizing

It demonstrates the platform's major capabilities as a coherent end-to-end engineering walkthrough, while keeping subsystem boundaries explicit rather than artificially coupling unrelated operations into a single execution:
1. **Knowledge Retrieval**: Ingesting a system architecture specification for *Project Titan* (distributed telemetry system) and performing hybrid search for capacity sizing formulas.
2. **Multi-Agent Orchestration**: The Supervisor analyzes the task and routes to `DataAgent`, which evaluates required ingestion worker nodes using an AST-based arithmetic tool (`3600 peak events/sec / 300 events/sec per node = 12 nodes`).
3. **Long-Term Memory**: Storing the sizing finding in semantic memory under scope `infra_audit` and proving scope isolation against unauthorized queries (`finance_audit`).
4. **Human-in-the-Loop (HITL)**: Attempting production deployment to `Site Alpha` triggers an explicit `interrupt()` safety gate, pausing the graph until an operator inspects and approves the action.
5. **Observability**: Extracting hierarchical OpenTelemetry spans and proving telemetry redaction of sensitive prompts.
6. **Evaluation**: Calculating quantitative retrieval metrics (`Recall@K`, `Precision@K`, `MRR`, context overlap) against benchmark ground truth.
7. **Time-Travel Replay**: Forking the execution thread to model a revised requirement (4,800 events/sec = 16 nodes) with mock tool overrides while keeping the source checkpoint immutable.

---

## 3. Canonical Architecture Flow

```mermaid
flowchart TD
    User([Task: Audit Project Titan & Size Capacity]) --> Setup[1. Environment Init]

    subgraph RAG Pipeline ["2. Knowledge Ingestion & Hybrid RAG"]
        Spec[system_spec.md] --> Ingest[Document Ingestion]
        Ingest --> ChromaStore[(Chroma KB)]
        Query[Query: Peak Capacity] --> HybridSearch[Hybrid Retrieval<br/>Dense + BM25 + RRF]
        ChromaStore --> HybridSearch
    end

    subgraph MultiAgentLoop ["3. LangGraph Orchestration & Guarded Tools"]
        HybridSearch --> Sup[Supervisor Agent]
        Sup -->|Route: data| DA[Data Agent]
        DA -->|Invoke: '3600 / 300'| Calc[Calculator Tool<br/>(AST Guardrail)]
        Calc -->|Result: 12.0| DA
        DA --> Sup
        Sup -->|Route: final| FA[Final Agent]
    end

    subgraph MemoryTier ["4. Semantic Memory Tier"]
        FA --> MemStore[Store: 12 Workers Needed<br/>Scope: infra_audit]
        MemStore --> ProbeMatch[Search: infra_audit -> Match]
        MemStore --> ProbeIso[Search: finance_audit -> Zero Matches]
    end

    subgraph SafetyGate ["5. Human-in-the-Loop (HITL)"]
        FA --> GateCheck{Target: Site Alpha?}
        GateCheck -->|Yes| Interrupt[LangGraph interrupt<br/>State Frozen in Checkpointer]
        Interrupt --> Operator[Operator Approval: POST /hitl/resume]
        Operator --> Resume[Resume & Synthesize]
    end

    subgraph TelemetryEvalReplay ["6-8. Telemetry, Eval & Replay"]
        Resume --> OTel[6. OpenTelemetry Audit<br/>21 Spans Captured & Redacted]
        OTel --> Eval[7. Offline Eval Metrics<br/>Recall@K: 1.0, MRR: 1.0]
        Eval --> Replay[8. Checkpoint Fork & Replay<br/>Override: 4800 QPS -> 16 Nodes]
    end

    Replay --> Complete([Demo Complete: Exit 0])
```

---

## 4. Prerequisites & Requirements

- **Python**: 3.11, 3.12, or 3.13 (Python 3.12 recommended).
- **Virtual Environment**: Active virtual environment with project dependencies installed (`requirements-dev.txt`).
- **External Network / APIs**: **None**. Operates 100% offline via `MockLLMProvider` and `MockEmbeddingProvider`.
- **Docker / PostgreSQL**: **Not required**. Operates in-memory via `MemorySaver` and ephemeral Chroma collections.

---

## 5. Execution Command

Run the canonical demo using the single CLI entrypoint:

```bash
# On Linux / macOS
python scripts/run_demo.py

# On Windows (PowerShell)
.venv\Scripts\python.exe scripts/run_demo.py
```

### Verbose Mode
To view internal application logs alongside the stage diagnostics:
```bash
python scripts/run_demo.py --verbose
```

---

## 6. Stage-by-Stage Breakdown

| Stage | Subsystem | Actions & Verifications |
|---|---|---|
| **[1/8] Environment** | Configuration | Initializes `Settings` with `llm_provider=mock`, ephemeral storage, and in-memory OpenTelemetry exporter. |
| **[2/8] Knowledge** | Hybrid RAG | Ingests `demo/data/knowledge/system_spec.md`, performs dense + BM25 hybrid retrieval via Reciprocal Rank Fusion ($\alpha=0.6$). |
| **[3/8] Orchestration** | LangGraph | Supervisor routes to `DataAgent`, invokes AST `calculator` with `3600 / 300`, verifies tool budget guardrails, and routes to `FinalAgent`. |
| **[4/8] Memory** | Chroma Semantic Store | Records capacity finding into `infra_audit` scope, verifies composite ranking retrieval, and probes `finance_audit` to prove cross-scope segregation. |
| **[5/8] HITL** | LangGraph Interrupt | Sensitive deployment action triggers `ApprovalGate`, pauses execution with checkpointer snapshot, inspects pending state, and resumes with operator approval. |
| **[6/8] Observability** | OpenTelemetry | Verifies in-process span hierarchy (root HTTP/orchestration spans, child agent spans), in-memory span collection, and audits redaction of sensitive prompts. |
| **[7/8] Evaluation** | Benchmark Metrics | Calculates mathematical retrieval metrics: `Recall@K=1.0`, `Precision@K=1.0`, `MRR=1.0`, and context-assertion overlap. |
| **[8/8] Replay** | ReplayService | Forks thread state from checkpoint, applies tool override (`4800 / 300 = 16`), proves new thread ID creation while source checkpoint remains immutable. |

---

## 7. Expected Console Output

```
====================================================================
 MULTI-AGENT AI ORCHESTRATION PLATFORM -- CANONICAL DEMONSTRATION
====================================================================
 Mode: Deterministic Local (100% Offline, Deterministic Workflow Verification)
 Target Scenario: Project Titan Architecture Audit & Capacity Sizing

[1/8] Environment & Runtime Initialization
      LLM Provider Mode                   : mock (Deterministic Mock) [PASS]
      Working Memory Checkpointer         : memory (MemorySaver) [PASS]
      Vector & Memory Storage             : Ephemeral Isolated Chroma DB [PASS]
      OpenTelemetry Instrumentation       : In-Memory Span Collector Active [PASS]

[2/8] Knowledge Ingestion & Hybrid RAG Retrieval
      Document Ingested                   : system_spec.md (id: ..., chunks indexed) [PASS]
      Hybrid Search Query                 : 'Peak burst capacity and worker throu...' [PASS]
      Top Match & Citation                : system_spec.md (RRF Score: 0.0325) [PASS]

[3/8] Multi-Agent Orchestration & Guarded Tool Execution
      Thread ID                           : demo_thread_titan_... [PASS]
      Supervisor Routing                  : Intent evaluated -> routed to 'data' agent [PASS]
      Guarded Tool Invocation             : calculator('3600 / 300') -> Result: 12.0 [PASS]
      Agents Participated                 : data, final [PASS]
      Execution Status                    : completed [PASS]
      Duration                            : 0.012s [PASS]

[4/8] Long-Term Semantic Memory & Cross-Scope Isolation
      Stored Memory ID                    : ... (scope: infra_audit) [PASS]
      Scope Search ('infra_audit')        : 1 match (Composite Score: 0.325) [PASS]
      Scope Isolation Probe ('finance_audit'): 0 matches (Zero leakage verified) [PASS]

[5/8] Human-in-the-Loop (HITL) Interruption & Approval
      Safety Gate Trigger                 : Target: Site Alpha (sensitive infrastructure) [PASS]
      LangGraph Interrupt Status          : interrupted (State persisted in checkpointer) [PASS]
      HITL Pending Inspection             : GET /api/v1/hitl/pending -> has_pending: True [PASS]
      Operator Decision Submitted         : Decision: 'approve' with architect feedback [PASS]
      Post-Approval Execution             : Status: completed [PASS]

[6/8] OpenTelemetry Tracing & Telemetry Audit
      Spans Captured in Trace             : 21 spans recorded in hierarchy [PASS]
      Root Span & Trace ID                : POST /api/v1/orchestration/run (trace: ...) [PASS]
      Span Operations Observed            : llm.call, supervisor.decide_route, hitl.interrupt... [PASS]
      Telemetry Redaction Audit           : Zero raw prompt or secret leakage in span attributes [PASS]

[7/8] Offline Evaluation Framework & Retrieval Quality
      Recall@K (k=3)                      : 1.000 [PASS]
      Precision@K (k=3)                   : 1.000 [PASS]
      Mean Reciprocal Rank (MRR)          : 1.000 [PASS]
      Context-Assertion Overlap           : 0.556 [PASS]

[8/8] Time-Travel Checkpoint Replay & Thread Fork
      Source Thread ID                    : demo_thread_titan_... (Checkpoint Immutable) [PASS]
      Forked Child Thread ID              : ... (Isolated Branch) [PASS]
      Tool Override Applied               : calculator mock result -> 16.0 workers [PASS]
      Replayed Outcome Answer             : Updated capacity requires 16 ingestion workers [PASS]
      Source State Immutability           : Original task intact: 'Audit Project Titan: calculate requi...' [PASS]

====================================================================
 CANONICAL DEMONSTRATION COMPLETE -- ALL 8 STAGES PASSED
 Total Duration: ~0.3 seconds | 100% Offline, Deterministic Workflow Verification
====================================================================
```

---

## 8. Determinism & Verification Boundaries

Workflow decisions, fixture data, retrieval ground truth, tool results, and expected invariants are deterministic; runtime identifiers, trace IDs, and execution duration are intentionally variable.

### What IS Verified
- Multi-agent state machine transitions and conditional edges in LangGraph.
- AST math parsing and tool execution boundaries.
- Hybrid lexical/dense document indexing and reciprocal rank fusion.
- Chroma memory persistence, scope querying, and zero cross-scope leakage.
- LangGraph `interrupt()` pausing, state serialization, and `Command(resume=...)` resumption.
- OpenTelemetry span hierarchy creation, parent-child links, and attribute redaction.
- Exact retrieval metric computation (`Recall@K`, `Precision@K`, `MRR`).
- Checkpoint restoration, branching (`fork=True`), and state override application.

### What is NOT Verified by This Demo
- **Live LLM Model Output Quality**: The demo uses the deterministic mock provider to guarantee zero-cost offline reproducibility without depending on external provider uptime or API rates.
- **Physical Docker Volume Remounts**: The demo executes in-memory. Container restart persistence is verified separately when Docker daemon is operational.
- **Production High-Throughput Load**: The script verifies functional correctness and architecture integration, not concurrency benchmarks.

---

## 9. Cleanup & Persistence

- The canonical demo runs with `ephemeral=True` for vector stores and in-memory checkpointing.
- No files are written to host disk during the demo.
- No database tables or persistent states are altered.
- Re-running the demo is completely idempotent and safe.
