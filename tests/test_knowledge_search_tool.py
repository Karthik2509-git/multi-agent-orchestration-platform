"""Tests for KnowledgeSearchTool."""

import pytest

from src.app.core.config import Settings
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService
from src.app.services.agent_service import build_default_tool_registry
from src.app.tools.knowledge_search import KnowledgeSearchTool


@pytest.fixture
def mock_rag_service():
    """Create isolated RAGService with in-memory Chroma for tool tests."""
    settings = Settings(rag_enabled=True, rag_embedding_provider="mock")
    store = ChromaVectorStore(collection_name="test_tool_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    return RAGService(vector_store=store, embedding_provider=embedder, settings=settings)


@pytest.mark.asyncio
async def test_knowledge_search_tool_execution(mock_rag_service):
    """Test KnowledgeSearchTool successfully retrieves results."""
    await mock_rag_service.ingest_text(
        content="Antigravity uses asynchronous agent workers and hybrid retrieval.",
        filename="antigravity_specs.md",
    )

    tool = KnowledgeSearchTool(rag_service=mock_rag_service)
    assert tool.name == "knowledge_search"
    assert "query" in tool.parameters_schema["properties"]

    result = await tool.execute(query="antigravity asynchronous")
    assert result.success is True
    assert result.data is not None
    assert result.data["count"] >= 1
    assert result.data["results"][0]["source"] == "antigravity_specs.md"


@pytest.mark.asyncio
async def test_knowledge_search_tool_validation(mock_rag_service):
    """Test KnowledgeSearchTool validates missing query parameter."""
    tool = KnowledgeSearchTool(rag_service=mock_rag_service)
    result = await tool.execute(query="")
    assert result.success is False
    assert "required" in result.error.lower()


def test_knowledge_search_tool_registry_integration(mock_rag_service):
    """Test KnowledgeSearchTool gets registered into default tool registry."""
    settings = Settings(rag_enabled=True)
    registry = build_default_tool_registry(settings=settings, rag_service=mock_rag_service)
    assert registry.has_tool("knowledge_search")
    assert registry.has_tool("calculator")
    assert registry.has_tool("http_get")
