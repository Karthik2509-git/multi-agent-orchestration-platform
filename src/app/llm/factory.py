"""Factory for instantiating LLM providers based on configuration."""

from src.app.core.config import Settings
from src.app.llm.base import LLMProvider
from src.app.llm.providers.gemini import GeminiLLMProvider
from src.app.llm.providers.groq import GroqLLMProvider
from src.app.llm.providers.mock import MockLLMProvider
from src.app.llm.providers.openai import OpenAILLMProvider
from src.app.llm.providers.openrouter import OpenRouterLLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate and return the configured active LLM provider.

    Raises:
        ValueError: If the active provider lacks a required API key or is unknown.
    """
    provider_name = settings.llm_provider.lower().strip()

    if provider_name == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured in environment or settings. "
                "Please configure OPENROUTER_API_KEY to use the OpenRouter provider."
            )
        return OpenRouterLLMProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )

    if provider_name == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in environment or settings. "
                "Please configure GEMINI_API_KEY to use the Google Gemini provider."
            )
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )

    if provider_name == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured in environment or settings. "
                "Please configure GROQ_API_KEY to use the Groq provider."
            )
        return GroqLLMProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )

    if provider_name == "mock":
        return MockLLMProvider()

    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured in environment or settings. "
                "Please configure OPENAI_API_KEY to use the OpenAI provider."
            )
        return OpenAILLMProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider_name}'. "
        "Allowed providers: openrouter, gemini, groq, mock, openai"
    )
