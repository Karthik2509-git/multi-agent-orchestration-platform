"""Versioned LLM model pricing metadata and cost estimation.

NOTE: This is an engineering observability estimate, not a financial billing system.
Provider pricing may change, cached pricing tables may become stale, and token usage
reported during intermediate runs may differ from final billed invoices.
Free-tier accounts and volume discounts are not represented.
"""

from enum import Enum
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

MODEL_PRICING_TABLE_VERSION = "2026-09-01-v1"


class CostStatus(str, Enum):
    """Status indicating how reliably LLM cost was calculated."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class CostEstimate(BaseModel):
    """Strongly typed cost estimation data structure."""

    model_config = ConfigDict(frozen=True)

    estimated_cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated monetary cost in USD, or None if unknown/unavailable.",
    )
    cost_status: CostStatus = Field(
        description="Evaluation state for cost calculation (known, unknown, or unavailable).",
    )
    pricing_version: Optional[str] = Field(
        default=None,
        description="Version identifier of the pricing table used, if known.",
    )
    pricing_source: Optional[str] = Field(
        default="versioned_registry",
        description="Source of the pricing reference data.",
    )


# Pricing per 1M tokens (input_price_usd_per_1m, output_price_usd_per_1m)
MODEL_PRICING_REGISTRY: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("openrouter", "openai/gpt-4o-mini"): (0.15, 0.60),
    ("openrouter", "anthropic/claude-3-5-sonnet"): (3.00, 15.00),
    ("gemini", "gemini-2.5-flash"): (0.075, 0.30),
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
}


def calculate_llm_cost(
    provider: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    pricing_version: Optional[str] = None,
) -> CostEstimate:
    """Calculate nullable cost estimate based on versioned pricing registry.

    Formula:
        input_cost = prompt_tokens / 1_000_000 * input_price_per_1M
        output_cost = completion_tokens / 1_000_000 * output_price_per_1M
        estimated_cost_usd = input_cost + output_cost

    Returns CostEstimate with:
    - CostStatus.KNOWN and calculated estimated_cost_usd if model pricing and tokens are available.
    - CostStatus.UNKNOWN and estimated_cost_usd=None if provider/model is not in registry.
    - CostStatus.UNAVAILABLE and estimated_cost_usd=None if token counts are missing/None.
    """
    normalized_key = (provider.strip().lower(), model.strip().lower())
    target_version = pricing_version or MODEL_PRICING_TABLE_VERSION

    if normalized_key not in MODEL_PRICING_REGISTRY:
        return CostEstimate(
            estimated_cost_usd=None,
            cost_status=CostStatus.UNKNOWN,
            pricing_version=None,
            pricing_source="versioned_registry",
        )

    if prompt_tokens is None or completion_tokens is None:
        return CostEstimate(
            estimated_cost_usd=None,
            cost_status=CostStatus.UNAVAILABLE,
            pricing_version=None,
            pricing_source="versioned_registry",
        )

    input_price_per_1m, output_price_per_1m = MODEL_PRICING_REGISTRY[normalized_key]
    input_cost = (prompt_tokens / 1_000_000.0) * input_price_per_1m
    output_cost = (completion_tokens / 1_000_000.0) * output_price_per_1m
    total_cost = round(input_cost + output_cost, 8)

    return CostEstimate(
        estimated_cost_usd=total_cost,
        cost_status=CostStatus.KNOWN,
        pricing_version=target_version,
        pricing_source="versioned_registry",
    )
