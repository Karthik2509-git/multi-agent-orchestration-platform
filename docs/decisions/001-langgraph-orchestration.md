# ADR 001: LangGraph for Multi-Agent Orchestration

## Status
Accepted

## Context
Multi-agent architectures require coordinating diverse LLM personas (supervisors, researchers, analysts, code generators) across iterative loops. Complex tasks demand branching, dynamic plan revisions, cycle limits, and deterministic state tracking. Traditional linear chains (e.g. basic sequential pipelines) lack cyclical control, while unconstrained autonomous loops risk infinite loops and unpredictable behavior. Furthermore, long-running agent workflows require checkpointing to support human interruption and recovery from failure.

## Decision
Adopt **LangGraph** (`StateGraph`) as the core workflow orchestration engine.
- Model multi-agent collaboration as a directed cyclical graph with a central `SupervisorAgent` routing to specialized workers (`research`, `data`, `code`, `final`).
- Maintain a single, strongly-typed `OrchestrationState` passed between graph nodes.
- Use LangGraph checkpointers (`AsyncPostgresSaver` in production, `MemorySaver` in development) to persist state snapshots after each step.
- Bound iteration execution using an explicit loop counter (`MAX_ITERATIONS = 10`).

## Consequences

### Positive
- **State Determinism**: Every agent transition operates on typed Pydantic/TypedDict state.
- **Durable Checkpointing**: Workflows can pause, persist to PostgreSQL, survive container crashes, and resume.
- **Native Interrupts**: Supports Human-in-the-Loop workflows directly via LangGraph's `interrupt()` primitive.
- **Traceability**: Clear node-by-node execution graphs that map directly onto OpenTelemetry spans.

### Negative / Trade-Offs
- Higher architectural complexity than single-agent ReAct loops.
- Requires thread management and unique `thread_id` tracking for state persistence.
- Dependency on LangGraph framework APIs.

## Architectural Comparison
- **Autogen / CrewAI**: Emphasize conversational emergence and conversational agent chat rooms. While flexible, they offer less deterministic control over exact state schemas, transitions, and state persistence compared to LangGraph's explicit graph topology.
- **Custom State Machine**: Would have required hand-crafting graph cycle detection, checkpointer persistence adapters, and interrupt/resume semantics from scratch.
