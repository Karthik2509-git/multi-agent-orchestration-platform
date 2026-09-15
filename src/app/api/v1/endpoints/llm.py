"""LLM provider informational and management endpoints."""

from fastapi import APIRouter, Depends, status

from src.app.core.config import Settings, get_settings
from src.app.models.schemas.provider import ProviderDetail, ProviderStatusResponse

router = APIRouter()


@router.get(
    "/providers",
    response_model=ProviderStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="List LLM Providers",
    description="Inspect supported LLM providers, active selection, and configuration readiness.",
)
async def get_providers_status(
    settings: Settings = Depends(get_settings),
) -> ProviderStatusResponse:
    """Return status and configured models for all supported providers without exposing secrets."""
    providers_info = {
        "openrouter": ProviderDetail(
            configured=bool(settings.openrouter_api_key),
            model=settings.openrouter_model,
        ),
        "gemini": ProviderDetail(
            configured=bool(settings.gemini_api_key),
            model=settings.gemini_model,
        ),
        "groq": ProviderDetail(
            configured=bool(settings.groq_api_key),
            model=settings.groq_model,
        ),
        "mock": ProviderDetail(
            configured=True,
            model="mock-model",
        ),
        "openai": ProviderDetail(
            configured=bool(settings.openai_api_key),
            model=settings.openai_model,
        ),
    }

    return ProviderStatusResponse(
        active_provider=settings.llm_provider,
        providers=providers_info,
    )
