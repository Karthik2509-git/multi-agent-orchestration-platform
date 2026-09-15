"""Schemas for LLM provider status and metadata."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ProviderDetail(BaseModel):
    """Configuration status details for a single LLM provider."""

    configured: bool = Field(
        description="Whether this provider has the required credentials configured"
    )
    model: Optional[str] = Field(
        default=None, description="Configured model identifier for this provider"
    )


class ProviderStatusResponse(BaseModel):
    """Response schema exposing LLM provider availability without leaking credentials."""

    active_provider: str = Field(description="Currently active LLM provider")
    providers: Dict[str, ProviderDetail] = Field(
        description="Configuration status and model info for all supported providers"
    )
