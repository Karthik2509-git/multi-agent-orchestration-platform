# ADR 005: Observability, Guardrails, Evaluation and Replay Architecture

## Status
Accepted

## Context
Multi-agent systems exhibit nondeterministic behaviors, multi-step compounding latencies, and variable token costs. Without end-to-end tracing and granular metrics, identifying which agent stalled, which tool failed, or where tokens were consumed becomes impossible. Furthermore, debugging failures requires reproducing historical execution states without re-running expensive or irreversible external actions. Finally, regression testing demands offline evaluation frameworks to benchmark agent performance across code changes.

## Decision
Implement a unified observability, safety guardrail, evaluation, and execution replay architecture:

1. **Distributed Tracing (OpenTelemetry)**:
   - Instrument the platform using OpenTelemetry (`src/app/observability/`).
   - Create hierarchical, context-propagating spans:
     `api.orchestration.run` → `orchestration.step.<n>` → `agent.<name>` → `tool.<name>` / `llm.generate`.
   - Record duration, agent metadata, and token volumes on spans.

2. **Cost & Metrics Accounting**:
   - Calculate prompt and completion token expenditures using model-specific pricing tables.
   - Collect in-memory system metrics (request rates, latency histograms, tool failure counters).

3. **Defensive Tool Guardrails**:
   - Enforce execution ceilings: 10 tool calls per run budget, consecutive-failure tool disablement (disabled for remainder of run after 3 consecutive failures), 64 KB input and 1 MB output payload limits.
   - Isolate tool errors so exceptions do not abort the orchestration loop.

4. **Time-Travel Checkpoint Replay (`ReplayService`)**:
   - Restore historical thread states directly from the checkpointer.
   - Allow execution forking (`fork=True`) to branch alternative paths from intermediate nodes.
   - Permit injecting mock tool results with strict schema validation.

5. **Offline Evaluation Framework (`src/app/evaluation/`)**:
   - Implement benchmark dataset runner evaluating agent accuracy, faithfulness, and tool selection without live production traffic.

## Consequences

### Positive
- **Deterministic Time-Travel Debugging**: Engineers can inspect, fork, and replay any failed execution step with modified parameters.
- **Cost Transparency**: Exact token costs are tracked and associated with individual agents and overall workflows.
- **Safe Blast Radius**: Run-scoped tool failure isolation and payload caps prevent runaway recursions and memory exhaustion.
- **Standardized Tracing**: Compatible with standard OpenTelemetry collectors and tracing backends (such as Jaeger or Datadog) via standard exporters.

### Negative / Trade-Offs
- Span creation and metric updates introduce slight runtime processing overhead.
- Maintaining pricing tables requires periodic updates when provider token rates change.

## Architectural Comparison
- **Proprietary Observability SDKs (e.g., LangSmith, Arize)**: Vendor lock-in limits deployment flexibility. Using OpenTelemetry ensures the platform remains vendor-neutral and portable across self-hosted and cloud observability stacks.
- **Log-Based Debugging**: Traditional text logs lack structured parent-child relationship graphs and cannot recreate historical graph state for re-execution.
