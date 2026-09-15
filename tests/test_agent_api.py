"""Tests for the Agent REST API endpoints."""

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall


def test_agent_run_endpoint_validation_empty_task(client: TestClient) -> None:
    """Test that empty task payloads are rejected with 422 Unprocessable Entity."""
    response = client.post("/api/v1/agent/run", json={"task": ""})
    assert response.status_code == 422



def test_agent_run_endpoint_missing_api_key(client: TestClient) -> None:
    """Test that agent endpoint returns 503 if OPENAI_API_KEY is not configured."""
    with patch("src.app.services.agent_service.get_llm_provider") as mock_factory:
        mock_factory.side_effect = ValueError(
            "OPENAI_API_KEY is not configured in environment or settings."
        )
        response = client.post("/api/v1/agent/run", json={"task": "Hello"})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "OPENAI_API_KEY is not configured" in response.json()["detail"]


def test_agent_run_endpoint_success_with_mock_provider(client: TestClient) -> None:
    """Test successful agent run endpoint execution using MockLLMProvider."""
    mock_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_calc",
                        name="calculator",
                        arguments={"expression": "15 * 10"},
                    )
                ],
                model="mock-gpt-4o",
            ),
            LLMResponse(
                content="The result of 15 * 10 is 150.",
                tool_calls=[],
                model="mock-gpt-4o",
            ),
        ]
    )

    with patch(
        "src.app.services.agent_service.get_llm_provider",
        return_value=mock_provider,
    ):
        response = client.post(
            "/api/v1/agent/run",
            json={"task": "Calculate 15 * 10"},
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["task"] == "Calculate 15 * 10"
        assert "150" in data["answer"]
        assert data["status"] == "completed"
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_name"] == "calculator"
        assert data["tool_calls"][0]["success"] is True
        assert data["tool_calls"][0]["result"]["result"] == 150
        assert data["execution_time_seconds"] >= 0
