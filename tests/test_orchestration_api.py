"""Tests for the Multi-Agent Orchestration REST API endpoint."""

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse


def test_orchestration_run_endpoint_validation_empty_task(client: TestClient) -> None:
    """Test that empty task payloads are rejected with 422 Unprocessable Entity."""
    response = client.post("/api/v1/orchestration/run", json={"task": ""})
    assert response.status_code == 422


def test_orchestration_run_endpoint_missing_api_key(client: TestClient) -> None:
    """Test that orchestration endpoint returns 503 if LLM provider is misconfigured."""
    with patch("src.app.services.orchestration_service.get_llm_provider") as mock_factory:
        mock_factory.side_effect = ValueError(
            "GEMINI_API_KEY is not configured in environment or settings."
        )
        response = client.post("/api/v1/orchestration/run", json={"task": "Orchestrate something"})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "GEMINI_API_KEY is not configured" in response.json()["detail"]


def test_orchestration_run_endpoint_success(client: TestClient) -> None:
    """Test successful multi-agent orchestration execution via REST API."""
    mock_provider = MockLLMProvider(
        responses=[
            # 1. Supervisor routes to data
            LLMResponse(
                content='{"next_agent": "data", "reasoning": "Compute statistical growth"}'
            ),
            # 2. Data agent runs
            LLMResponse(content="Data analysis computed growth rate at 22%."),
            # 3. Supervisor routes to final
            LLMResponse(
                content='{"next_agent": "final", "reasoning": "Data calculation complete"}'
            ),
            # 4. Final agent produces final answer
            LLMResponse(content="The computed growth rate for the given dataset is 22%."),
        ]
    )

    with patch(
        "src.app.services.orchestration_service.get_llm_provider",
        return_value=mock_provider,
    ):
        response = client.post(
            "/api/v1/orchestration/run",
            json={"task": "Analyze the dataset growth rate"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["task"] == "Analyze the dataset growth rate"
        assert "22%" in data["answer"]
        assert data["status"] == "completed"
        assert "data" in data["agents_used"]
        assert "final" in data["agents_used"]
        assert data["execution_time_seconds"] >= 0
