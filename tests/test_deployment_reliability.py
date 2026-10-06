"""Tests for Phase 8 Milestone 2: Deployment reliability and self-healing readiness."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from scripts.verify_deployment import run_verification
from src.app.core.config import Settings, get_settings
from src.app.main import app
from src.app.memory.working_memory import (
    check_postgres_readiness,
    reset_checkpointer,
    set_connection_pool,
)


def _create_mock_pool() -> MagicMock:
    """Create a mock AsyncConnectionPool with an async context manager for connection()."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = None

    class MockContextManager:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_pool.connection.return_value = MockContextManager()
    return mock_pool


@pytest.mark.asyncio
async def test_readiness_uses_existing_pool_without_reinitialization() -> None:
    """When pool is already initialized, readiness checks use it directly without re-init."""
    mock_pool = _create_mock_pool()
    set_connection_pool(mock_pool, MagicMock())

    postgres_settings = Settings(
        app_name="Test App",
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="test_password",
    )

    try:
        with patch("src.app.memory.working_memory.init_checkpointer") as mock_init:
            is_ready = await check_postgres_readiness(settings=postgres_settings)
            assert is_ready is True
            mock_init.assert_not_called()
            mock_pool.connection.assert_called_once()
    finally:
        reset_checkpointer()


@pytest.mark.asyncio
async def test_readiness_self_heals_when_postgres_becomes_available() -> None:
    """When pool is missing and PostgreSQL becomes reachable, readiness self-heals."""
    reset_checkpointer()

    postgres_settings = Settings(
        app_name="Test App",
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="test_password",
    )

    mock_pool = _create_mock_pool()

    async def mock_init(settings=None):
        set_connection_pool(mock_pool, MagicMock())
        return MagicMock()

    try:
        with patch(
            "src.app.memory.working_memory.init_checkpointer",
            side_effect=mock_init,
        ) as patched_init:
            is_ready = await check_postgres_readiness(settings=postgres_settings)
            assert is_ready is True
            patched_init.assert_called_once()
            mock_pool.connection.assert_called_once()
    finally:
        reset_checkpointer()


@pytest.mark.asyncio
async def test_readiness_fails_safely_when_postgres_remains_unavailable() -> None:
    """When pool is missing and PostgreSQL is down, readiness fails safely and returns False."""
    reset_checkpointer()

    postgres_settings = Settings(
        app_name="Test App",
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="test_password",
    )

    try:
        with patch(
            "src.app.memory.working_memory.init_checkpointer",
            side_effect=RuntimeError("Connection refused: postgres:5432"),
        ):
            is_ready = await check_postgres_readiness(settings=postgres_settings)
            assert is_ready is False
    finally:
        reset_checkpointer()


@pytest.mark.asyncio
async def test_concurrent_readiness_requests_prevent_competing_initializations() -> None:
    """Concurrent readiness checks while pool is uninitialized execute initialization only once."""
    reset_checkpointer()

    postgres_settings = Settings(
        app_name="Test App",
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="test_password",
    )

    mock_pool = _create_mock_pool()
    init_call_count = 0

    async def delayed_init(settings=None):
        nonlocal init_call_count
        init_call_count += 1
        # Small delay to simulate connection pool spin-up and ensure concurrency overlap
        await asyncio.sleep(0.02)
        set_connection_pool(mock_pool, MagicMock())
        return MagicMock()

    try:
        with patch(
            "src.app.memory.working_memory.init_checkpointer",
            side_effect=delayed_init,
        ):
            # Launch 5 concurrent readiness checks simultaneously
            tasks = [check_postgres_readiness(settings=postgres_settings) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All 5 concurrent probes should succeed
            assert all(results)
            # init_checkpointer must only have executed ONCE due to the initialization lock
            assert init_call_count == 1
    finally:
        reset_checkpointer()


@pytest.mark.asyncio
async def test_memory_checkpoint_backend_bypasses_postgres_readiness() -> None:
    """In-memory checkpointer mode bypasses PostgreSQL readiness checks completely."""
    reset_checkpointer()

    memory_settings = Settings(
        app_name="Test App",
        app_env="testing",
        checkpoint_backend="memory",
    )

    with patch("src.app.memory.working_memory.init_checkpointer") as mock_init:
        is_ready = await check_postgres_readiness(settings=memory_settings)
        assert is_ready is True
        mock_init.assert_not_called()


def test_liveness_remains_independent_and_unaffected() -> None:
    """GET /health remains fast and returns 200 regardless of checkpointer or PostgreSQL state."""
    reset_checkpointer()

    postgres_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="test_password",
    )

    app.dependency_overrides[get_settings] = lambda: postgres_settings
    try:
        with patch("src.app.memory.working_memory.init_checkpointer") as mock_init:
            with TestClient(app) as test_client:
                response = test_client.get("/health")
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["status"] == "healthy"
                mock_init.assert_not_called()
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()


def test_api_v1_health_endpoint_self_heals_to_200() -> None:
    """GET /api/v1/health successfully self-heals when PostgreSQL becomes available."""
    reset_checkpointer()

    postgres_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        checkpoint_backend="postgres",
        postgres_password="test_password",
    )

    mock_pool = _create_mock_pool()

    async def mock_init(settings=None):
        set_connection_pool(mock_pool, MagicMock())
        return MagicMock()

    app.dependency_overrides[get_settings] = lambda: postgres_settings
    try:
        with patch(
            "src.app.memory.working_memory.init_checkpointer",
            side_effect=mock_init,
        ):
            with TestClient(app) as test_client:
                response = test_client.get("/api/v1/health")
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["status"] == "healthy"
    finally:
        reset_checkpointer()
        app.dependency_overrides.clear()


def test_verify_deployment_script_unreachable_endpoint() -> None:
    """verify_deployment script cleanly reports failure when target service is unreachable."""
    # Running verification against an unused local port should return exit code 1 cleanly
    exit_code = run_verification(base_url="http://127.0.0.1:59999", timeout=0.5)
    assert exit_code == 1


def test_verify_deployment_script_success_with_mock_provider() -> None:
    """verify_deployment script successfully validates all 5 stages using mock provider."""
    mock_settings = Settings(
        app_name="Multi-Agent Orchestration Platform Test",
        app_env="testing",
        app_version="0.1.0-test",
        llm_provider="mock",
        checkpoint_backend="memory",
        rag_embedding_provider="mock",
        rag_persist_directory=None,
        memory_persist_directory=None,
    )

    app.dependency_overrides[get_settings] = lambda: mock_settings
    try:
        with TestClient(app) as test_client:
            exit_code = run_verification(base_url="http://testserver", client=test_client)
            assert exit_code == 0
    finally:
        app.dependency_overrides.clear()
