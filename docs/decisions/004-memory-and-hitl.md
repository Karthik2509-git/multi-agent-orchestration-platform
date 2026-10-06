# ADR 004: Dual-Tier Memory and Human-in-the-Loop Architecture

## Status
Accepted

## Context
Autonomous multi-agent workflows require two distinct types of state:
1. **Immediate Execution State**: The transient step-by-step history, tool scratchpads, and execution graphs of an ongoing task.
2. **Long-Term Knowledge State**: Facts, user preferences, and organizational constraints that must persist across disparate conversations and threads.

Furthermore, autonomous agents operating in production environments must not blindly execute high-consequence actions or make decisions when uncertainty is high. The system requires a mechanism to halt execution, persist intermediate state, await human review, and resume seamlessly.

## Decision
Implement a **Dual-Tier Memory System** combined with **State-Driven Human-in-the-Loop (HITL)**:

1. **Working Memory (Execution State)**:
   - Handled exclusively via **LangGraph Checkpointing** (`AsyncPostgresSaver` in production, `MemorySaver` in development).
   - Serializes complete graph state after every node transition, indexed by `thread_id`.

2. **Long-Term Semantic Memory (Persistent Knowledge)**:
   - Implemented as an isolated Chroma collection (`agent_memory`).
   - Partitioned by scope (`user`, `agent`, `session`, `global`).
   - Extracted using entity/constraint heuristics, deduplicated via cosine similarity ($>0.85$), and ranked via composite scoring:
     $$\text{Score} = 0.65 \times \text{Similarity} + 0.25 \times \text{Importance} + 0.10 \times \text{Recency}$$
   - Automatically injected into the Supervisor Agent's reasoning context prior to task execution.

3. **Human-in-the-Loop (HITL)**:
   - An `ApprovalGate` node evaluates step proposals against safety policies and supervisor routing confidence ($<0.65$).
   - Sensitive actions trigger LangGraph's native `interrupt()`, halting progression and writing pending action details to the checkpointer.
   - Operations expose dedicated REST endpoints:
     - `GET /api/v1/hitl/pending/{thread_id}`: Read pending action state.
     - `POST /api/v1/hitl/resume/{thread_id}`: Unpause execution via `Command(resume=...)`.

## Consequences

### Positive
- **Clear Separation of Concerns**: Working state and long-term memories do not pollute each other.
- **Resilient Asynchronous Review**: A workflow can remain paused for minutes, hours, or days in PostgreSQL without holding memory threads or server connections open.
- **Auditability**: All pending approvals, decisions, and reviewer feedback are permanently recorded in thread checkpoints.

### Negative / Trade-Offs
- Managing two storage engines (PostgreSQL for checkpoints, Chroma for vector memories) requires separate health checks and volume configurations.
- Resuming interrupted threads requires external clients to track and query by `thread_id`.

## Architectural Comparison
- **Unified Monolithic Database**: Storing graph checkpoints directly inside a vector database like Chroma is inefficient, as graph execution requires strict relational transactional semantics (ACID) provided by PostgreSQL.
- **In-Memory Thread Blocking**: Pausing execution using Python thread locks or active HTTP connections would cause timeouts and crash if the backend container restarted during review.
