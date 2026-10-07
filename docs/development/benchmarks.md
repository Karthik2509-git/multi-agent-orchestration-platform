# Platform Benchmarks & Engineering Evidence
> Local benchmark runs write machine-specific metadata and variable timing results to `benchmarks/results/local/benchmark_report.json` by default. This directory is Git-ignored so running the benchmark does not overwrite the committed reference report.


This document establishes the benchmarking architecture, experimental methodologies, reproducibility protocols, and claim boundaries for the **Multi-Agent AI Orchestration Platform**.

---

## 1. Purpose

The objective of the platform benchmark suite is to provide transparent, scientifically grounded engineering evidence rather than synthetic marketing numbers. Instead of collapsing heterogeneous performance metrics into a single arbitrary score, the suite cleanly separates:

1. **Deterministic Retrieval Quality**: Evaluated on versioned, reproducible benchmark fixtures.
2. **Local Framework Overhead**: Micro-benchmarked across host-level components under local execution.
3. **Correctness & Safety Invariants**: Deterministically verified guardrail boundaries and tenant isolation.
4. **Token-Cost Projections**: Calculated via a versioned pricing registry under canonical workflow assumptions.

---

## 2. Benchmark Architecture

The benchmark runner (`scripts/run_benchmarks.py`) executes five decoupled suites sequentially within an isolated, deterministic execution context:

```text
scripts/run_benchmarks.py
 ├── Suite 1: RAG Retrieval Quality (RAGEvaluator + 10 Fixture Cases)
 ├── Suite 2: Local Framework Overhead (perf_counter_ns, N=30 iterations)
 ├── Suite 3: Tool Budget & Failure Isolation (ToolExecutionContext)
 ├── Suite 4: Multi-Tenant Memory Scope Segregation (ChromaMemoryStore)
 └── Suite 5: Token / Cost Modeling Projections (MODEL_PRICING_REGISTRY)
      │
      ▼
 benchmarks/results/local/benchmark_report.json
```

---

## 3. Reproduction Command

The complete benchmark suite executes 100% offline with deterministic evaluation/invariant suites and variable local runtime measurements, requiring no external network connectivity, paid API keys, or external service dependencies.

```bash
# Standard run (N=30 iterations, 3 warmup iterations)
python scripts/run_benchmarks.py

# Custom sample size and explicit output destination
python scripts/run_benchmarks.py --iterations 50 --warmup 5 --output-json benchmarks/results/benchmark_report.json

# Verbose execution with full internal trace logging
python scripts/run_benchmarks.py --verbose
```

---

## 4. Environment Metadata

Every execution automatically records reproducibility metadata into the generated local report (by default `benchmarks/results/local/benchmark_report.json`):

| Field | Description | Example Value |
|---|---|---|
| `benchmark_version` | Semantic version of the benchmark suite | `1.0.0` |
| `git_commit` | Source control commit SHA of the tested code | `2c339bebb4bbef445a609ffbd0e06f23fa938fc6` |
| `python_version` | Host Python runtime version | `3.14.3` |
| `platform` | Host operating system and CPU architecture | `Windows-11-10.0.26200-SP0` |
| `execution_mode` | Runtime environment classification | `100% offline execution with deterministic evaluation/invariant suites and variable local runtime measurements` |
| `iterations` | Number of measured iterations per latency operation | `30` |
| `warmup_iterations` | Warm-up runs executed prior to timing | `3` |
| `timestamp_utc` | ISO 8601 UTC timestamp of execution | `2026-10-06T20:12:06Z` |

---

## 5. Suite 1 Methodology: RAG Retrieval Quality

### Fixture Dataset
Reuses the platform's deterministic evaluation infrastructure (`get_deterministic_benchmark_dataset()` in [dataset.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/multi-agent-orchestration-platform/src/app/evaluation/dataset.py)). The dataset comprises 10 hand-curated test cases representing specific retrieval challenge modes:
- Rank-1 exact match
- Rank-3 reciprocal ranking
- Empty relevance ground truth (relevance negative control)
- Empty retrieval returns
- Multi-document relevant queries
- Fewer retrieved documents than $K$
- Duplicated document deduplication
- Lexically grounded assertion overlap
- Ungrounded assertion zero-overlap

### Evaluator & Metrics
Evaluated using [evaluator.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/multi-agent-orchestration-platform/src/app/evaluation/evaluator.py) at rank threshold $K=5$:
- **Recall@5**: Proportion of relevant documents successfully retrieved in top-$K$ ($0.6667$).
- **Precision@5**: Proportion of retrieved documents that are relevant ($0.3817$).
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first relevant document ($0.6333$).
- **Context Assertion Overlap**: Mean token overlap ratio between retrieved context and ground-truth assertions ($0.6673$).

> [!NOTE]
> The 10 test cases represent deterministic engineering fixtures for retrieval regression prevention. They must not be cited as a generalized real-world domain benchmark. Context assertion overlap serves as a lexical token-overlap containment proxy, not semantic entailment or Natural Language Inference (NLI).

---

## 6. Suite 2 Methodology: Local Framework Processing Latency

### Targeted Components
Measures local in-process framework processing overhead across four core operations:
1. **Multi-Agent Orchestration Loop**: Full turn execution across supervisor routing, specialist dispatch, and mock provider synthesis.
2. **Hybrid RAG Retrieval**: In-memory BM25 lexical search, vector similarity search, and Reciprocal Rank Fusion ($k=60$).
3. **Semantic Memory Search**: In-memory Chroma embedding query and cosine similarity filtering.
4. **AST Calculator Tool**: Safe arithmetic AST parse, validation, and mathematical evaluation.

### Timing Protocol
- Measurement tool: `time.perf_counter_ns()`.
- Iteration protocol: 3 unmeasured warmup iterations followed by $N=30$ measured iterations.
- Reported statistics: `mean`, `p50` (median), `p90`, `p99`, `min`, `max` in milliseconds (ms).

---

## 7. Suite 3 Methodology: Tool Budget & Failure Isolation

Suite 3 deterministically tests the run-scoped execution boundaries enforced by [execution_context.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/multi-agent-orchestration-platform/src/app/tools/execution_context.py):

1. **Tool Budget Invariant**: An execution context configured with a budget of 10 permits exactly 10 tool invocations and strictly rejects the 11th invocation with `ToolBudgetExceededError`.
2. **Consecutive Failure Isolation**: A failing tool is allowed 3 consecutive errors. Upon the 3rd failure, the tool is disabled for the remainder of the run; subsequent invocations are rejected with `ToolExecutionError`.
3. **Streak Reset**: A successful tool execution immediately resets the consecutive failure counter to 0, ensuring intermittent errors do not prematurely disable active tools.

---

## 8. Suite 4 Methodology: Multi-Tenant Memory Scope Segregation

Suite 4 verifies strict isolation between distinct tenant scopes within [chroma_memory_store.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/multi-agent-orchestration-platform/src/app/memory/stores/chroma_memory_store.py):

- Two distinct scopes are provisioned: `tenant_infra_core` (Scope A) and `tenant_billing_audit` (Scope B).
- Memories with unique canary tokens (`CANARY_ALPHA_*` and `CANARY_BETA_*`) are stored under each scope.
- Cross-scope search queries are executed (querying Scope B using Scope A search terms and canaries, and vice versa).
- **Observed Result**: $0$ cross-scope data leaks ($0.0\%$ leakage rate). All target queries retrieve only their respective tenant's memories.

> [!NOTE]
> This represents deterministic verification of zero cross-scope leakage across the tested isolation scenarios under Chroma metadata filtering. It should not be framed as a universal mathematical proof of the entire memory subsystem.

---

## 9. Suite 5 Methodology: Token Accounting & Cost Projections

Suite 5 demonstrates deterministic token-cost projections using the platform's versioned pricing registry ([cost.py](file:///c:/Users/KARTHIK%20V/OneDrive/Desktop/multi-agent-orchestration-platform/src/app/llm/cost.py), version `2026-09-01-v1`):

- **Representative Canonical Workload Assumption**: Cost projections use a fixed representative 1,070-token workload (850 prompt + 220 completion tokens distributed across supervisor routing, specialist tool invocation, and final synthesis) to compare the configured pricing models. No provider billing or live token usage is involved.
- **Projected Costs per 1,000 Invocations**:
  - `gemini/gemini-2.5-flash`: **$0.1298**
  - `openai/gpt-4o-mini`: **$0.2595**
  - `groq/llama-3.3-70b-versatile`: **$0.6753**
  - `openrouter/anthropic/claude-3-5-sonnet`: **$5.8500**

> [!NOTE]
> These figures represent deterministic token-cost projections derived from standard public rates in the registry under fixed representative workload assumptions. They do not represent actual billed enterprise expenditure or negotiated volume tiering.

---

## 10. Deterministic vs Variable Measurements

To prevent misleading benchmarking claims, the report explicitly segregates:

```text
Deterministic (Stable across runs)        Variable (Host/Runtime Dependent)
----------------------------------        ---------------------------------
- RAG Metric Scores (Recall, Precision)   - Multi-agent loop latency (ms)
- Context Assertion Overlap               - Hybrid search retrieval latency (ms)
- Tool Budget Limit Enforcement (10/11)   - Memory vector search latency (ms)
- Tool Consecutive Failure Disable (3)    - AST tool execution latency (ms)
- Cross-Scope Memory Leakage (0 leaks)    - Operating system scheduling jitter
- Token Calculation & Cost Projections
```

---

## 11. Interpretation of P50, P90, and P99

- **P50 (Median)**: The expected typical framework overhead for a standard request.
- **P90**: Baseline tail latency under normal local system load.
- **P99**: **Approximate upper-tail observation.** With $N=30$ sample iterations, $P99$ is inherently coarse. It should be interpreted as an indicative upper-tail observation rather than a production Service Level Objective (SLO) estimate.

---

## 12. Claim Boundaries

The following claim boundaries are strictly enforced across platform documentation:
- **No Production Performance Claims**: Local micro-benchmarks measure Python framework overhead and in-process execution on the local development host. They do not represent production platform throughput, end-to-end user-facing response times, or network-bound response times. Never present local processing times as "platform latency" or "AI agent response time".
- **No Real LLM Latency Claims**: Benchmark runs use deterministic mock providers and in-memory stores; external LLM latency (typically 500ms–3000ms) is deliberately excluded.
- **No Entailment Claims**: Lexical context-assertion overlap is a token-containment heuristic and does not guarantee semantic natural language entailment.
- **Empirical Isolation Verification**: Memory isolation evidence demonstrates zero leakage across defined test scenarios, not an unconstrained mathematical guarantee.
- **Cost Projections Only**: Financial models reflect pricing registry calculations based on fixed representative workload assumptions rather than historical cloud invoice records.

---

## 13. Known Limitations

1. **Local Host Dependency**: Host CPU speed, memory bandwidth, and background OS scheduling cause run-to-run variations in Suite 2 latency measurements.
2. **Coarse Sample Tail ($N=30$)**: While sufficient for local development smoke-testing and regression detection, production SLO characterization requires $N \ge 1,000$ in a dedicated staging environment.
3. **In-Memory Store Semantics**: Tests execute against SQLite and in-memory Chroma instances rather than clustered distributed persistence.

---

## 14. What Is Explicitly NOT Benchmarked

To preserve engineering integrity, the platform deliberately excludes:
- **Live Paid LLM API Invocations**: Avoids unpredictable network jitter, vendor rate-limiting, and unnecessary expenditure.
- **Synthetic Scaled Millions-of-Vectors Load**: Avoids artificial Chroma scaling tests unrepresentative of agentic working memory.
- **Synthetic QPS / Throughput Load Tests**: Prevents conflating local single-process loop performance with distributed load capacity.
- **Docker / Virtualization Overhead**: Measures pure application runtime characteristics.
- **LLM-as-a-Judge Accuracy**: Avoids non-deterministic, cost-prohibitive automated judge metrics.
- **Microsecond Security Guardrails**: Measures correctness boundaries rather than micro-benchmarking regex filters.
