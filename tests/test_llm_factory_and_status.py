"""Tests for LLM factory selection and provider status endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app.core.config import Settings
from src.app.llm.factory import get_llm_provider
from src.app.llm.providers.gemini import GeminiLLMProvider
from src.app.llm.providers.groq import GroqLLMProvider
from src.app.llm.providers.mock import MockLLMProvider
from src.app.llm.providers.openrouter import OpenRouterLLMProvider


def test_factory_selects_mock() -> None:
    """Test that mock provider is selected without requiring API keys."""
    settings = Settings(llm_provider="mock")
    provider = get_llm_provider(settings)
    assert isinstance(provider, MockLLMProvider)


def test_factory_selects_openrouter() -> None:
    """Test selecting OpenRouter provider when configured."""
    settings = Settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test",
        openrouter_model="openrouter/free",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenRouterLLMProvider)
    assert provider.model == "openrouter/free"


def test_factory_selects_gemini() -> None:
    """Test selecting Gemini provider when configured."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="gemini-key-test",
        gemini_model="gemini-2.5-flash",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.model == "gemini-2.5-flash"


def test_factory_selects_groq() -> None:
    """Test selecting Groq provider when configured."""
    settings = Settings(
        llm_provider="groq",
        groq_api_key="groq-key-test",
        groq_model="llama-3.3-70b-versatile",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GroqLLMProvider)
    assert provider.model == "llama-3.3-70b-versatile"


def test_factory_active_provider_missing_key_raises() -> None:
    """Test that if the active provider is missing its key, ValueError is raised."""
    settings = Settings(llm_provider="gemini", gemini_api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        get_llm_provider(settings)

    settings_or = Settings(llm_provider="openrouter", openrouter_api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        get_llm_provider(settings_or)

    settings_groq = Settings(llm_provider="groq", groq_api_key="")
    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        get_llm_provider(settings_groq)


def test_factory_inactive_missing_keys_do_not_fail() -> None:
    """Test that missing keys for inactive providers do NOT cause errors."""
    settings = Settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-or-valid-key",
        gemini_api_key="",
        groq_api_key="",
        openai_api_key="",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenRouterLLMProvider)


def test_factory_unknown_provider_raises() -> None:
    """Test that unsupported provider names are rejected."""
    # Bypass Pydantic field validator using model_construct to test factory safety
    settings = Settings.model_construct(llm_provider="unsupported_ai")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm_provider(settings)


def test_provider_status_endpoint(client: TestClient) -> None:
    """Test GET /api/v1/llm/providers returns configuration without leaking secrets."""
    response = client.get("/api/v1/llm/providers")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "active_provider" in data
    assert "providers" in data

    providers = data["providers"]
    assert "openrouter" in providers
    assert "gemini" in providers
    assert "groq" in providers
    assert "mock" in providers

    assert isinstance(providers["mock"]["configured"], bool)
    assert providers["mock"]["configured"] is True

    # Ensure zero secrets in response
    json_text = response.text.lower()
    assert "api_key" not in json_text
    assert "secret" not in json_text
    assert "authorization" not in json_text
