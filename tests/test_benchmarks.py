"""Automated validation tests for Benchmark Suite and Engineering Evidence (Phase 8 M5).

Verifies benchmark report schema, metric ranges, invariance properties,
and execution reliability without brittle assertions on wall-clock latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.run_benchmarks import (
    BENCHMARK_VERSION,
    run_all_benchmarks,
    run_suite_1_rag_quality,
    run_suite_2_latency,
    run_suite_3_tool_guardrails,
    run_suite_4_memory_isolation,
    run_suite_5_cost_projections,
)


@pytest.mark.asyncio
async def test_suite_1_rag_quality_metrics() -> None:
    """Validate deterministic RAG retrieval metrics on benchmark fixtures."""
    res = await run_suite_1_rag_quality()
    assert "mean_recall_at_5" in res
    assert "mean_precision_at_5" in res
    assert "mean_mrr" in res
    assert "mean_context_assertion_overlap" in res

    assert 0.0 <= res["mean_recall_at_5"] <= 1.0
    assert 0.0 <= res["mean_precision_at_5"] <= 1.0
    assert 0.0 <= res["mean_mrr"] <= 1.0
    assert 0.0 <= res["mean_context_assertion_overlap"] <= 1.0
    assert res["total_cases"] == 10
    assert len(res["cases"]) == 10


@pytest.mark.asyncio
async def test_suite_2_latency_structure() -> None:
    """Validate latency distributional characterization structure (smoke test N=2)."""
    res = await run_suite_2_latency(iterations=2, warmup_iterations=1)
    assert res["iterations"] == 2
    assert "operations" in res

    expected_ops = [
        "multi_agent_orchestration",
        "hybrid_rag_retrieval",
        "semantic_memory_search",
        "ast_calculator_tool",
    ]
    for op in expected_ops:
        assert op in res["operations"]
        stats = res["operations"][op]
        assert stats["mean_ms"] >= 0.0
        assert stats["p50_ms"] >= 0.0
        assert stats["p90_ms"] >= 0.0
        assert stats["p99_ms"] >= 0.0
        assert stats["min_ms"] >= 0.0
        assert stats["max_ms"] >= 0.0
        assert stats["min_ms"] <= stats["max_ms"]


@pytest.mark.asyncio
async def test_suite_3_tool_guardrails_invariants() -> None:
    """Validate run-scoped tool budget and consecutive-failure disablement invariants."""
    res = await run_suite_3_tool_guardrails()
    assert res["all_invariants_passed"] is True
    assert res["budget_limit_enforced_11th_blocked"] is True
    assert res["consecutive_failure_disablement_enforced"] is True
    assert res["disabled_tool_remains_blocked"] is True
    assert res["streak_reset_on_success_verified"] is True
    assert res["calls_permitted_before_budget"] == 10


@pytest.mark.asyncio
async def test_suite_4_memory_scope_isolation_invariants() -> None:
    """Validate zero cross-scope memory leakage across test isolation scenarios."""
    res = await run_suite_4_memory_isolation()
    assert res["all_invariants_passed"] is True
    assert res["cross_scope_leakage_count"] == 0
    assert res["cross_scope_leakage_rate"] == 0.0
    assert res["scope_a_target_hits"] >= 1
    assert res["scope_b_target_hits"] >= 1
    assert "Deterministic verification" in res["claim"]


def test_suite_5_cost_projections_structure() -> None:
    """Validate deterministic token-cost projections using the pricing registry."""
    res = run_suite_5_cost_projections()
    assert res["pricing_registry_version"] == "2026-09-01-v1"
    assert "assumed_token_consumption" in res
    assert res["assumed_token_consumption"]["total_tokens"] > 0
    assert "projections" in res
    assert len(res["projections"]) >= 4

    for model_key, proj in res["projections"].items():
        assert proj["cost_status"] == "known"
        assert proj["projected_cost_usd_per_run"] >= 0.0
        assert proj["projected_cost_usd_per_1k_runs"] >= 0.0


@pytest.mark.asyncio
async def test_run_all_benchmarks_end_to_end(tmp_path: Path) -> None:
    """Execute complete benchmark runner with temporary output and verify report schema."""
    output_path = tmp_path / "benchmark_test_output.json"
    exit_code = await run_all_benchmarks(
        iterations=2,
        warmup_iterations=1,
        output_json_path=output_path,
        verbose=False,
    )
    assert exit_code == 0
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Top-level sections
    assert data["benchmark_version"] == BENCHMARK_VERSION
    assert "metadata" in data
    assert "deterministic_results" in data
    assert "variable_runtime_results" in data
    assert "correctness_invariants" in data
    assert "cost_projections" in data

    # Metadata fields
    meta = data["metadata"]
    assert "git_commit" in meta
    assert "python_version" in meta
    assert "platform" in meta
    assert "execution_mode" in meta
    assert "iterations" in meta
    assert "warmup_iterations" in meta
    assert "timestamp_utc" in meta

    # Deterministic RAG
    rag = data["deterministic_results"]["rag_retrieval_quality"]
    assert 0.0 <= rag["mean_recall_at_5"] <= 1.0
    assert 0.0 <= rag["mean_precision_at_5"] <= 1.0
    assert 0.0 <= rag["mean_mrr"] <= 1.0

    # Invariants
    invs = data["correctness_invariants"]
    assert invs["tool_budget_and_failure_isolation"]["all_invariants_passed"] is True
    assert invs["memory_scope_isolation"]["all_invariants_passed"] is True
    assert invs["memory_scope_isolation"]["cross_scope_leakage_rate"] == 0.0
