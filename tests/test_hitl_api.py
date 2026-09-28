from unittest.mock import patch

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings, get_settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.main import create_app
from src.app.models.schemas.llm import LLMResponse


def test_hitl_api_pause_and_resume_flow():
    """Verify task pauses with pending status and resumes successfully through HITL API."""
    app = create_app()
    checkpointer = MemorySaver()
    settings = Settings(
        checkpoint_backend="memory",
        memory_enabled=False,
        rag_embedding_provider="mock",
    )

    mock_llm = MockLLMProvider(
        responses=[
            # Step 1: Supervisor routes to final
            LLMResponse(content='{"next_agent": "final"}', model="mock-llm"),
            # Step 2: Final agent produces response on resume
            LLMResponse(content="Final approved response via API.", model="mock-llm"),
        ]
    )

    app.dependency_overrides[get_settings] = lambda: settings
    thread_id = "test_hitl_api_thread_1"

    with (
        patch(
            "src.app.services.orchestration_service.get_llm_provider",
            return_value=mock_llm,
        ),
        patch(
            "src.app.services.orchestration_service.get_checkpointer",
            return_value=checkpointer,
        ),
        patch(
            "src.app.api.v1.endpoints.hitl.get_llm_provider",
            return_value=mock_llm,
        ),
        patch(
            "src.app.api.v1.endpoints.hitl.get_checkpointer",
            return_value=checkpointer,
        ),
    ):
        with TestClient(app) as client:
            # 1. Trigger task that requires human approval
            run_resp = client.post(
                "/api/v1/orchestration/run",
                json={
                    "task": "Execute sensitive API deletion",
                    "thread_id": thread_id,
                    "require_human_review": True,
                },
            )
            assert run_resp.status_code == 200
            run_data = run_resp.json()
            assert run_data["status"] == "interrupted"
            assert run_data["thread_id"] == thread_id
            assert run_data["pending_approval"] is not None

            # 2. Inspect pending approval
            pending_resp = client.get(f"/api/v1/hitl/pending/{thread_id}")
            assert pending_resp.status_code == 200
            pending_data = pending_resp.json()
            assert pending_data["has_pending_approval"] is True
            assert pending_data["request"]["thread_id"] == thread_id

            # 3. Resume with APPROVE
            resume_resp = client.post(
                f"/api/v1/hitl/resume/{thread_id}",
                json={
                    "decision": "approve",
                    "feedback": "Confirmed by QA engineer.",
                },
            )
            assert resume_resp.status_code == 200
            resume_data = resume_resp.json()
            assert resume_data["status"] == "completed"
            assert "Final approved response" in resume_data["answer"]
