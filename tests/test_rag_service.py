"""Tests for RAGService coordination."""

import pytest

from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService


@pytest.fixture
def rag_service():
    """Create an isolated test RAGService with Mock embeddings and in-memory Chroma."""
    settings = Settings(
        rag_enabled=True,
        rag_embedding_provider="mock",
        rag_persist_directory=None,
    )
    vector_store = ChromaVectorStore(
        collection_name="test_service_kb",
        ephemeral=True,
    )
    embedder = MockEmbeddingProvider(dimension=16)
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(
                content="LangGraph is a framework for stateful multi-agent workflows [Source 1].",
                model="mock-llm",
            )
        ]
    )
    return RAGService(
        vector_store=vector_store,
        embedding_provider=embedder,
        llm_provider=mock_llm,
        settings=settings,
    )


@pytest.mark.asyncio
async def test_ingest_and_retrieve_text(rag_service):
    """Test ingesting text and retrieving chunks."""
    doc = await rag_service.ingest_text(
        content="LangGraph enables circular graph workflows for LLM multi-agent systems.",
        filename="langgraph_intro.txt",
        metadata={"author": "LangChain Team"},
    )
    assert doc.id is not None
    assert doc.filename == "langgraph_intro.txt"

    results = await rag_service.retrieve(query="circular graph workflows", top_k=2)
    assert len(results) >= 1
    assert "LangGraph" in results[0].content


@pytest.mark.asyncio
async def test_query_with_grounded_citations(rag_service):
    """Test end-to-end question answering pipeline with citation extraction."""
    await rag_service.ingest_text(
        content="FastAPI is a modern web framework with automatic OpenAPI docs.",
        filename="fastapi_doc.md",
        metadata={"category": "backend"},
    )

    response = await rag_service.query(query_text="What is FastAPI?")
    assert response["query"] == "What is FastAPI?"
    assert response["answer"] is not None
    assert len(response["citations"]) >= 1
    assert response["citations"][0]["source"] == "fastapi_doc.md"


@pytest.mark.asyncio
async def test_list_and_delete_documents(rag_service):
    """Test listing documents and deleting by document ID."""
    doc = await rag_service.ingest_text(
        content="Temporary document content to test deletion.",
        filename="temp_delete.txt",
    )

    docs_before = await rag_service.list_documents()
    assert len(docs_before) >= 1

    deleted_count = await rag_service.delete_document(doc.id)
    assert deleted_count >= 1

    docs_after = await rag_service.list_documents()
    assert all(d["document_id"] != doc.id for d in docs_after)


@pytest.mark.asyncio
async def test_get_stats(rag_service):
    """Test retrieving knowledge base statistics."""
    stats = await rag_service.get_stats()
    assert "total_chunks" in stats
    assert "total_documents" in stats
    assert "embedding_provider" in stats
    assert stats["embedding_provider"] == "MockEmbeddingProvider"
