"""Tests for health check endpoints."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient


def test_root_health_endpoint(client: TestClient) -> None:
    """Test that GET /health returns 200 and expected payload structure."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "Multi-Agent Orchestration Platform Test"
    assert data["version"] == "0.1.0-test"
    assert data["environment"] == "testing"
    assert "timestamp" in data


def test_api_v1_health_endpoint(client: TestClient) -> None:
    """Test that GET /api/v1/health returns 200 and identical payload structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "Multi-Agent Orchestration Platform Test"
    assert data["version"] == "0.1.0-test"
    assert data["environment"] == "testing"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_async_health_endpoint(async_client: AsyncClient) -> None:
    """Test health endpoint asynchronously using httpx."""
    response = await async_client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"


def test_cors_headers_allowed_origin(client: TestClient) -> None:
    """Test that CORS headers are correctly returned for configured allowed origins."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/health", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_headers_disallowed_origin(client: TestClient) -> None:
    """Test that preflight requests from unauthorized origins are rejected."""
    headers = {
        "Origin": "http://malicious-site.com",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/health", headers=headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
