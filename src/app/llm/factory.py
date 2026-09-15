"""Factory for instantiating LLM providers."""

from src.app.core.config import Settings
from src.app.llm.base import LLMProvider
from src.app.llm.providers.openai import OpenAILLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate and return the configured LLM provider."""
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured in environment or settings. "
            "Please configure OPENAI_API_KEY to use the live OpenAI provider."
        )
    return OpenAILLMProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
