# Platform Benchmarks & Engineering Evidence

This directory contains the reproducible benchmark report and results schema for the Multi-Agent AI Orchestration Platform.

---

## Directory Structure

```text
benchmarks/
├── README.md                           # Directory guide and reproduction instructions
└── results/
    └── benchmark_report.json           # Canonical machine-readable benchmark artifact
```

---

## Reproduction Command

The complete benchmark suite executes 100% offline with deterministic evaluation/invariant suites and variable local runtime measurements, requiring no external API keys or cloud dependencies:

```bash
# Standard 30-iteration distributional characterization
python scripts/run_benchmarks.py

# Custom iterations and custom output path
python scripts/run_benchmarks.py --iterations 50 --warmup 5 --output-json benchmarks/results/benchmark_report.json
```

---

## Benchmark Philosophy & Claim Boundaries

The platform separates experimental results into distinct, principled evidence categories:

1. **Deterministic Retrieval Quality (`deterministic_results`)**:
   - Evaluated across curated, reproducible benchmark fixtures (`rag_deterministic_benchmark_v1`).
   - Groundedness metric (`context_assertion_overlap`) represents lexical token-overlap containment, not semantic entailment / NLI.
   - Fixtures represent deterministic engineering test cases, not a generalized real-world domain benchmark.

2. **Variable Local Processing Latency (`variable_runtime_results`)**:
   - Characterizes in-process framework orchestration, vector distance search, and AST tool parsing overhead under local host execution.
   - **Excludes external LLM inference and wide-area network latency.**
   - $N=30$ measured iterations provide lightweight local distributional characterization; $P99$ is an approximate upper-tail observation and should not be interpreted as a production Service Level Objective (SLO).

3. **Deterministic Safety Invariants (`correctness_invariants`)**:
   - Verifies run-scoped tool budget limits (10 permitted, 11th blocked) and consecutive-failure tool disablement (3 failures threshold, reset on success).
   - Verifies zero cross-scope memory leakage ($0.0\%$) across tested multi-tenant isolation scenarios.

4. **Token-Cost Projections (`cost_projections`)**:
   - Projections calculated deterministically from the versioned pricing registry (`MODEL_PRICING_REGISTRY`).
   - Cost projections use a fixed representative 1,070-token workload (850 prompt + 220 completion tokens) to compare the configured pricing models. No provider billing or live token usage is involved.

---

## Benchmark Report Metadata

Every generated `benchmark_report.json` captures complete reproducibility context:
- `git_commit`: Current commit SHA from which measurements were collected.
- `python_version`: Host Python runtime.
- `platform`: Host operating system and processor architecture.
- `execution_mode`: Verified offline mode identifier (`100% offline execution with deterministic evaluation/invariant suites and variable local runtime measurements`).
- `iterations` & `warmup_iterations`: Sample size configuration.
- `timestamp_utc`: Generation timestamp.
