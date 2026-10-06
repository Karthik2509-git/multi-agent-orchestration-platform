# Architecture Diagrams

This document collects architectural and sequence diagrams illustrating key platform subsystems.

---

## 1. Multi-Agent Orchestration Topology

```mermaid
flowchart TD
    Start([User Request]) --> Sup[Supervisor Agent]

    subgraph Routing ["Routing & Evaluation"]
        Sup --> Decision{Next Action?}
        Decision -->|Research Needed| R[Research Agent]
        Decision -->|Computation Needed| D[Data Agent]
        Decision -->|Code Analysis Needed| C[Code Agent]
        Decision -->|Sensitive / Low Confidence| AG[Approval Gate Node]
        Decision -->|All Goals Met| F[Final Agent]
    end

    subgraph Specialist Execution
        R --> TR[Tool Registry]
        D --> TR
        C -->|Static Analysis Only| Done[No Tools]
        TR --> Native[Native Tools / RAG]
        TR --> MCP[MCP Adapter]
    end

    subgraph State Feedback
        R --> Sup
        D --> Sup
        C --> Sup
        AG -->|Human Resumed| Sup
    end

    F --> EndNode([Final Answer])
```

---

## 2. Hybrid RAG Architecture (Dense + BM25 + RRF)

```mermaid
flowchart LR
    Query[Search Query] --> Parallel{Fan-Out}

    subgraph Dense Stream
        Parallel --> Embed[ONNX Embeddings<br/>all-MiniLM-L6-v2]
        Embed --> Chroma[Chroma Vector Store<br/>Cosine Similarity]
        Chroma --> DenseRank[Ranked Dense List]
    end

    subgraph Lexical Stream
        Parallel --> Tokenize[Tokenization / Stemming]
        Tokenize --> BM25[BM25 Index<br/>Frequency Scoring]
        BM25 --> LexRank[Ranked Lexical List]
    end

    DenseRank --> RRF[Reciprocal Rank Fusion<br/>RRF Score = 0.6 Dense + 0.4 Lexical]
    LexRank --> RRF
    RRF --> TopK[Top-K Fused Chunks]
```

---

## 3. Dual-Tier Memory Architecture

```mermaid
flowchart TB
    subgraph Working Memory ["Working Memory (Transient / Checkpointed)"]
        W1[Current Step State]
        W2[Agent Scratchpad]
        W3[Thread Execution History]
        W1 & W2 & W3 --> PostgresSaver["AsyncPostgresSaver / MemorySaver<br/>(Graph Checkpointer)"]
    end

    subgraph Long-Term Memory ["Long-Term Semantic Memory (Persistent)"]
        RawText[Execution Text / Output] --> Extractor[Memory Extractor<br/>Facts, Preferences, Constraints]
        Extractor --> Dedup{Cosine Sim > 0.85?}
        Dedup -->|Yes| Merge[Merge / Update Record]
        Dedup -->|No| Insert[Insert New Vector]
        Merge & Insert --> ChromaMem["Chroma Store<br/>(Collection: agent_memory)"]

        ChromaMem --> Ranker["Composite Scorer<br/>0.65 Sim + 0.25 Imp + 0.10 Recency"]
        Ranker --> ContextInject[Supervisor Memory Context]
    end

    WorkingMemory <--> OrchestrationGraph
    ContextInject --> OrchestrationGraph
```

---

## 4. Human-in-the-Loop (HITL) Interrupt & Resume Cycle

```mermaid
sequenceDiagram
    autonumber
    participant App as Orchestration Graph
    participant Gate as ApprovalGate Node
    participant DB as Postgres Checkpointer
    participant API as FastAPI /hitl Endpoints
    actor Operator as Human Reviewer

    App->>Gate: Evaluate step action & confidence
    Note over Gate: Action is sensitive OR confidence < 0.65
    Gate->>DB: Persist state & set pending_action
    Gate->>App: invoke interrupt()
    App-->>API: 200 OK (status: 'interrupted', thread_id)

    Operator->>API: GET /api/v1/hitl/pending/{thread_id}
    API-->>Operator: Pending action details & proposed params

    alt Approved
        Operator->>API: POST /api/v1/hitl/resume/{thread_id} (approved=True)
        API->>App: ainvoke(Command(resume={'approved': True}))
        App->>DB: Update state (approval_status: 'approved')
        App->>App: Proceed with tool execution
    else Rejected
        Operator->>API: POST /api/v1/hitl/resume/{thread_id} (approved=False, feedback='...')
        API->>App: ainvoke(Command(resume={'approved': False, 'feedback': '...'}))
        App->>DB: Update state (approval_status: 'rejected')
        App->>App: Route back to Supervisor with feedback
    end
```

---

## 5. Tool Guardrail & Consecutive-Failure Disablement Workflow

```mermaid
flowchart TD
    Req[Tool Invocation Request] --> SizeCheck{Input Payload <= 64 KB?}
    SizeCheck -->|No| ErrSize[Reject: Payload Size Exceeded]
    SizeCheck -->|Yes| BudgetCheck{Run Tool Calls < 10?}
    BudgetCheck -->|No| ErrBudget[Reject: Tool Budget Exhausted]
    BudgetCheck -->|Yes| FailCheck{Consecutive Failures < 3?}
    FailCheck -->|No| ErrDisabled[Reject: Tool Disabled for Remainder of Run]
    FailCheck -->|Yes| Exec[Execute Tool Logic]

    Exec --> SuccessCheck{Execution Succeeded?}
    SuccessCheck -->|Yes| ResetCounter[Reset Failure Count to 0]
    ResetCounter --> OutputCheck{Output Payload <= 1 MB?}
    OutputCheck -->|Yes| ReturnSuccess[Return ToolResult]
    OutputCheck -->|No| TruncateOutput[Truncate Payload & Warn]

    SuccessCheck -->|No| IncCounter[Increment Failure Count]
    IncCounter --> ReturnIsolatedErr[Return Structured Error ToolResult]
```

---

## 6. Execution Replay & Fork Architecture

```mermaid
flowchart LR
    OriginalThread[(Original Checkpoint History)] --> Fetch[Replay Service]
    Fetch --> TargetStep{Target Node / Step}

    TargetStep --> ForkDecision{Fork New Thread?}
    ForkDecision -->|fork=True| NewThread[(New Thread Checkpoint)]
    ForkDecision -->|fork=False| InPlace[(Resume in Existing Thread)]

    Overrides[Override Inputs / Mock Tool Results] --> ApplyOverrides[Validate & Patch Checkpoint State]
    ApplyOverrides --> ResumeGraph[Resume LangGraph Execution]
    NewThread --> ApplyOverrides
    InPlace --> ApplyOverrides
    ResumeGraph --> Output[New Orchestration Response]
```
