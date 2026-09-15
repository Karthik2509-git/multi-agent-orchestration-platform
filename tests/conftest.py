"""Shared test fixtures for pytest."""

from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.app.core.config import Settings, get_settings
from src.app.main import app


def get_test_settings() -> Settings:
    """Provide deterministic test settings."""
    return Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        debug=True,
        log_level="CRITICAL",
        cors_origins=["http://testserver"],
    )


@pytest.fixture(scope="session")
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
