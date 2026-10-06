"""Unit tests for Phase 7 Milestone 2 versioned LLM cost tracking."""

import pytest

from src.app.observability.cost import (
    MODEL_PRICING_REGISTRY,
    MODEL_PRICING_TABLE_VERSION,
    CostEstimate,
    CostStatus,
    calculate_llm_cost,
)


def test_cost_status_contract_values():
    """Verify CostStatus enum defines exactly the conceptual states required."""
    assert CostStatus.KNOWN.value == "known"
    assert CostStatus.UNKNOWN.value == "unknown"
    assert CostStatus.UNAVAILABLE.value == "unavailable"


def test_pricing_registry_known_models():
    """Verify the approved initial models exist in the versioned pricing registry."""
    assert ("openrouter", "openai/gpt-4o-mini") in MODEL_PRICING_REGISTRY
    assert ("openrouter", "anthropic/claude-3-5-sonnet") in MODEL_PRICING_REGISTRY
    assert ("gemini", "gemini-2.5-flash") in MODEL_PRICING_REGISTRY
    assert ("groq", "llama-3.3-70b-versatile") in MODEL_PRICING_REGISTRY
    assert ("openai", "gpt-4o-mini") in MODEL_PRICING_REGISTRY


def test_cost_calculation_known_model():
    """Test 21 & 49: Known pricing produces a CostEstimate with correct status and version."""
    estimate = calculate_llm_cost(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert estimate.cost_status == CostStatus.KNOWN
    assert estimate.pricing_version == MODEL_PRICING_TABLE_VERSION
    assert estimate.pricing_source == "versioned_registry"
    # openai/gpt-4o-mini is 0.15 input / 0.60 output per 1M tokens
    assert estimate.estimated_cost_usd == pytest.approx(0.75, abs=1e-6)


def test_cost_calculation_mathematical_precision():
    """Test 44, 45, 46: Verify input, output, and combined cost calculation formulas."""
    # prompt: 200_000 tokens of openai/gpt-4o-mini (0.15/1M) = 0.030 USD
    # completion: 50_000 tokens of openai/gpt-4o-mini (0.60/1M) = 0.030 USD
    estimate = calculate_llm_cost(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=200_000,
        completion_tokens=50_000,
    )
    assert estimate.cost_status == CostStatus.KNOWN
    assert estimate.estimated_cost_usd == pytest.approx(0.060, abs=1e-6)


def test_cost_calculation_unknown_model_never_returns_zero():
    """Test 22 & 47: Unknown model produces None + CostStatus.UNKNOWN and never 0.0."""
    estimate = calculate_llm_cost(
        provider="unknown_provider",
        model="nonexistent-model",
        prompt_tokens=500,
        completion_tokens=200,
    )
    assert estimate.cost_status == CostStatus.UNKNOWN
    assert estimate.estimated_cost_usd is None
    assert estimate.estimated_cost_usd != 0.0
    assert estimate.pricing_version is None


def test_cost_calculation_missing_prompt_tokens():
    """Test 23 & 48: Missing prompt tokens produces None + CostStatus.UNAVAILABLE."""
    estimate = calculate_llm_cost(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=None,
        completion_tokens=100,
    )
    assert estimate.cost_status == CostStatus.UNAVAILABLE
    assert estimate.estimated_cost_usd is None
    assert estimate.estimated_cost_usd != 0.0


def test_cost_calculation_missing_completion_tokens():
    """Test 23 & 48: Missing completion tokens produces None + CostStatus.UNAVAILABLE."""
    estimate = calculate_llm_cost(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=None,
    )
    assert estimate.cost_status == CostStatus.UNAVAILABLE
    assert estimate.estimated_cost_usd is None
    assert estimate.estimated_cost_usd != 0.0


def test_cost_calculation_uses_configured_pricing_version():
    """Test 24: Cost calculation respects custom pricing version if supplied."""
    custom_version = "2026-12-01-v2"
    estimate = calculate_llm_cost(
        provider="groq",
        model="llama-3.3-70b-versatile",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing_version=custom_version,
    )
    assert estimate.cost_status == CostStatus.KNOWN
    assert estimate.pricing_version == custom_version
    # groq / llama-3.3-70b-versatile: 0.59 + 0.79 = 1.38
    assert estimate.estimated_cost_usd == pytest.approx(1.38, abs=1e-6)


def test_cost_calculation_case_insensitivity():
    """Verify provider and model names are normalized cleanly."""
    estimate = calculate_llm_cost(
        provider="  OPENAI  ",
        model="  GPT-4O-MINI  ",
        prompt_tokens=100_000,
        completion_tokens=100_000,
    )
    assert estimate.cost_status == CostStatus.KNOWN
    assert estimate.estimated_cost_usd is not None


def test_cost_estimate_model_immutability():
    """Verify CostEstimate model fields cannot be mutated in place."""
    estimate = CostEstimate(
        estimated_cost_usd=0.05,
        cost_status=CostStatus.KNOWN,
        pricing_version=MODEL_PRICING_TABLE_VERSION,
    )
    with pytest.raises(Exception):
        estimate.estimated_cost_usd = 0.10
