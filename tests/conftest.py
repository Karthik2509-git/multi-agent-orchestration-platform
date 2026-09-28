"""Shared test fixtures for pytest."""

import os
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.app.core.config import Settings, get_settings
from src.app.main import app

# Explicitly configure test environment defaults
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("CHECKPOINT_BACKEND", "memory")
get_settings.cache_clear()


def get_test_settings() -> Settings:
    """Provide deterministic test settings."""
    return Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        debug=True,
        log_level="CRITICAL",
        cors_origins=["http://testserver"],
        checkpoint_backend="memory",
        rag_embedding_provider="mock",
        rag_persist_directory=None,
        memory_persist_directory=None,
        memory_similarity_threshold=0.0,
    )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Synchronous test client fixture."""
    app.dependency_overrides[get_settings] = get_test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous test client fixture."""
    app.dependency_overrides[get_settings] = get_test_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()
