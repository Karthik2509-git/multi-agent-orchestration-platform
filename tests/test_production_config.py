"""Tests for production configuration validation and health/readiness endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.app.core.config import Settings, get_settings
from src.app.main import app
from src.app.memory.working_memory import reset_checkpointer, set_connection_pool


def test_production_placeholder_password_rejected() -> None:
    """Production environment must reject placeholder/default PostgreSQL passwords."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            app_env="production",
            checkpoint_backend="postgres",
            postgres_password="changeme_in_production",
        )
    errors = str(exc_info.value)
    assert "Insecure default POSTGRES_PASSWORD" in errors

    # Check other insecure placeholders
    with pytest.raises(ValidationError) as exc_info2:
        Settings(
            app_env="production",
            checkpoint_backend="postgres",
            postgres_password="postgres_dev_password",
        )
    assert "Insecure default POSTGRES_PASSWORD" in str(exc_info2.value)

    with pytest.raises(ValidationError) as exc_info3:
        Settings(
            app_env="production",
            checkpoint_backend="postgres",
            postgres_password="",
        )
    assert "Insecure default POSTGRES_PASSWORD" in str(exc_info3.value)


def test_production_strong_password_accepted() -> None:
    """Production environment accepts strong non-placeholder passwords."""
    prod_settings = Settings(
        app_env="production",
        checkpoint_backend="postgres",
        postgres_password="V3ry$tr0ngPr0duct10nP@ssw0rd!",
    )
    assert prod_settings.app_env == "production"
    assert prod_settings.postgres_password == "V3ry$tr0ngPr0duct10nP@ssw0rd!"


def test_development_placeholder_password_accepted() -> None:
    """Development and testing environments allow placeholder passwords for local setup."""
    dev_settings = Settings(
        app_env="development",
        checkpoint_backend="postgres",
        postgres_password="changeme_in_production",
    )
    assert dev_settings.app_env == "development"
    assert dev_settings.postgres_password == "changeme_in_production"

    test_settings = Settings(
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="changeme_in_production",
    )
    assert test_settings.app_env == "testing"
    assert test_settings.postgres_password == "changeme_in_production"


def test_readiness_postgres_success() -> None:
    """Readiness endpoint returns 200 when PostgreSQL is required and reachable."""
    postgres_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        checkpoint_backend="postgres",
        postgres_password="test_postgres_password",
    )

    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = None

    class MockContextManager:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_pool.connection.return_value = MockContextManager()

    set_connection_pool(mock_pool, MagicMock())
    try:
        app.dependency_overrides[get_settings] = lambda: postgres_settings
        with TestClient(app) as test_client:
            response = test_client.get("/api/v1/health")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "healthy"
            assert data["app_name"] == "Multi-Agent Orchestration Platform Test"
            assert data["environment"] == "testing"
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()


def test_readiness_postgres_unavailable_returns_503() -> None:
    """Readiness endpoint returns 503 when PostgreSQL is required but unreachable."""
    postgres_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        checkpoint_backend="postgres",
        postgres_password="test_postgres_password",
    )

    reset_checkpointer()
    app.dependency_overrides[get_settings] = lambda: postgres_settings
    try:
        with patch(
            "src.app.memory.working_memory.check_postgres_readiness",
            new=AsyncMock(return_value=False),
        ):
            with TestClient(app) as test_client:
                response = test_client.get("/api/v1/health")
                assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                data = response.json()
                assert data["status"] == "unhealthy"
                assert data["app_name"] == "Multi-Agent Orchestration Platform Test"
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()


def test_readiness_postgres_exception_returns_503() -> None:
    """Readiness endpoint returns 503 when PostgreSQL query raises an exception."""
    postgres_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        checkpoint_backend="postgres",
        postgres_password="test_postgres_password",
    )

    mock_pool = MagicMock()

    class FailingContextManager:
        async def __aenter__(self):
            raise ConnectionRefusedError("Database connection refused")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_pool.connection.return_value = FailingContextManager()
    set_connection_pool(mock_pool, MagicMock())
    try:
        app.dependency_overrides[get_settings] = lambda: postgres_settings
        with TestClient(app) as test_client:
            response = test_client.get("/api/v1/health")
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert data["status"] == "unhealthy"
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()


def test_memory_checkpoint_backend_does_not_require_postgres() -> None:
    """In-memory checkpointer mode does not check or require PostgreSQL for readiness."""
    memory_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        checkpoint_backend="memory",
    )

    reset_checkpointer()
    app.dependency_overrides[get_settings] = lambda: memory_settings
    try:
        with patch("src.app.memory.working_memory.check_postgres_readiness") as mock_pg_check:
            with TestClient(app) as test_client:
                response = test_client.get("/api/v1/health")
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["status"] == "healthy"
                mock_pg_check.assert_not_called()
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()


def test_root_liveness_independent_of_postgres() -> None:
    """Liveness probe GET /health returns 200 even when PostgreSQL is unavailable."""
    postgres_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        checkpoint_backend="postgres",
        postgres_password="test_postgres_password",
    )

    reset_checkpointer()
    app.dependency_overrides[get_settings] = lambda: postgres_settings
    try:
        with patch("src.app.memory.working_memory.check_postgres_readiness") as mock_pg_check:
            with TestClient(app) as test_client:
                response = test_client.get("/health")
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["status"] == "healthy"
                mock_pg_check.assert_not_called()
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()
