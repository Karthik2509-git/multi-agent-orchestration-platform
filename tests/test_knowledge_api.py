"""Tests for Knowledge & RAG REST API endpoints."""

import io

import pytest
from fastapi.testclient import TestClient

from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.main import create_app
from src.app.models.schemas.llm import LLMResponse
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService, get_rag_service


@pytest.fixture
def mock_rag_app():
    """Create test application with isolated RAGService dependency override."""
    settings = Settings(rag_enabled=True, rag_embedding_provider="mock")
    store = ChromaVectorStore(collection_name="test_api_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Grounded answer based on knowledge base [Source 1].",
                model="mock-llm",
            )
        ]
    )
    rag_service = RAGService(
        vector_store=store,
        embedding_provider=embedder,
        llm_provider=mock_llm,
        settings=settings,
    )

    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    return app, rag_service


def test_knowledge_text_ingestion_and_search(mock_rag_app):
    """Test POST /api/v1/knowledge/documents and POST /api/v1/knowledge/search."""
    app, rag_service = mock_rag_app
    with TestClient(app) as client:
        # Ingest document
        ingest_resp = client.post(
            "/api/v1/knowledge/documents",
            json={
                "content": "Antigravity is an AI agent orchestration system.",
                "filename": "overview.txt",
                "metadata": {"section": "architecture"},
            },
        )
        assert ingest_resp.status_code == 201
        data = ingest_resp.json()
        assert "document_id" in data
        assert data["filename"] == "overview.txt"

        # List documents
        list_resp = client.get("/api/v1/knowledge/documents")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] >= 1
        assert any(d["filename"] == "overview.txt" for d in list_data["documents"])

        # Search documents
        search_resp = client.post(
            "/api/v1/knowledge/search",
            json={
                "query": "orchestration system",
                "top_k": 3,
                "strategy": "rrf",
            },
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert search_data["count"] >= 1
        assert "Antigravity" in search_data["results"][0]["content"]


def test_knowledge_file_upload(mock_rag_app):
    """Test POST /api/v1/knowledge/documents/upload."""
    app, _ = mock_rag_app
    with TestClient(app) as client:
        file_content = b"# Architecture Overview\n\nModular design with vector search."
        files = {"file": ("architecture.md", io.BytesIO(file_content), "text/markdown")}
        upload_resp = client.post("/api/v1/knowledge/documents/upload", files=files)
        assert upload_resp.status_code == 201
        data = upload_resp.json()
        assert data["filename"] == "architecture.md"


def test_knowledge_grounded_query(mock_rag_app):
    """Test POST /api/v1/knowledge/query."""
    app, _ = mock_rag_app
    with TestClient(app) as client:
        # Ingest reference text
        client.post(
            "/api/v1/knowledge/documents",
            json={
                "content": "The platform supports LangGraph and MCP v2 protocols.",
                "filename": "protocols.txt",
            },
        )

        query_resp = client.post(
            "/api/v1/knowledge/query",
            json={
                "query": "What protocols are supported?",
                "top_k": 2,
            },
        )
        assert query_resp.status_code == 200
        data = query_resp.json()
        assert "answer" in data
        assert len(data["citations"]) >= 1
        assert data["citations"][0]["source"] == "protocols.txt"


def test_knowledge_stats_and_delete(mock_rag_app):
    """Test GET /api/v1/knowledge/stats and DELETE /api/v1/knowledge/documents/{id}."""
    app, _ = mock_rag_app
    with TestClient(app) as client:
        # Check stats
        stats_resp = client.get("/api/v1/knowledge/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert "total_chunks" in stats

        # Ingest and then delete
        ingest_resp = client.post(
            "/api/v1/knowledge/documents",
            json={"content": "Content to be deleted.", "filename": "to_delete.txt"},
        )
        doc_id = ingest_resp.json()["document_id"]

        del_resp = client.delete(f"/api/v1/knowledge/documents/{doc_id}")
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data["chunks_deleted"] >= 1

        # Delete non-existent
        not_found_resp = client.delete(f"/api/v1/knowledge/documents/{doc_id}")
        assert not_found_resp.status_code == 404
