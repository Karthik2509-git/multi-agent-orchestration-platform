"""Integration tests for Memory REST API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.app.main import create_app
from src.app.memory.service import MemoryService, get_memory_service
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore
from src.app.rag.embeddings import MockEmbeddingProvider


@pytest.fixture
def mock_memory_app():
    """Create FastAPI test application with ephemeral MemoryService dependency override."""
    app = create_app()
    embedder = MockEmbeddingProvider(dimension=16)
    store = ChromaMemoryStore(
        collection_name="test_api_memory_kb",
        embedding_provider=embedder,
        ephemeral=True,
    )
    memory_service = MemoryService(store=store)

    app.dependency_overrides[get_memory_service] = lambda: memory_service
    return app, memory_service


def test_memory_api_lifecycle(mock_memory_app):
    """Test full REST API lifecycle: create, get, list, search, consolidate, delete."""
    app, _ = mock_memory_app
    with TestClient(app) as client:
        # 1. Create Memory
        create_resp = client.post(
            "/api/v1/memory",
            json={
                "content": "User prefers JSON formatted outputs.",
                "scope_id": "test_user",
                "memory_type": "user_preference",
                "importance": 0.85,
            },
        )
        assert create_resp.status_code == 201
        data = create_resp.json()["memory"]
        memory_id = data["id"]
        assert data["content"] == "User prefers JSON formatted outputs."
        assert data["scope_id"] == "test_user"

        # 2. Get Memory
        get_resp = client.get(f"/api/v1/memory/{memory_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["memory"]["id"] == memory_id

        # 3. List Memories
        list_resp = client.get("/api/v1/memory?scope_id=test_user")
        assert list_resp.status_code == 200
        assert list_resp.json()["count"] == 1

        # 4. Search Memories
        search_resp = client.post(
            "/api/v1/memory/search",
            json={
                "query": "JSON format preference",
                "scope_id": "test_user",
                "top_k": 2,
            },
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert search_data["count"] >= 1
        assert search_data["results"][0]["memory"]["id"] == memory_id

        # 5. Stats
        stats_resp = client.get("/api/v1/memory/stats?scope_id=test_user")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_memories"] == 1

        # 6. Consolidate
        cons_resp = client.post("/api/v1/memory/consolidate?scope_id=test_user")
        assert cons_resp.status_code == 200
        assert cons_resp.json()["scope_id"] == "test_user"

        # 7. Delete Memory
        del_resp = client.delete(f"/api/v1/memory/{memory_id}?scope_id=test_user")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # 8. Verify Gone
        get_after = client.get(f"/api/v1/memory/{memory_id}")
        assert get_after.status_code == 404
