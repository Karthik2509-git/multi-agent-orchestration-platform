"""End-to-end integration tests for ToolCallingAgent using KnowledgeSearchTool."""

import pytest

from src.app.agents.tool_calling_agent import ToolCallingAgent
from src.app.core.config import Settings
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse, ToolCall
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService
from src.app.services.agent_service import build_default_tool_registry
from src.app.tools.knowledge_search import KnowledgeSearchTool
from src.app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_executes_knowledge_search_tool():
    """Verify ToolCallingAgent calls knowledge_search and incorporates knowledge in final answer."""
    # 1. Setup RAGService with indexed document
    settings = Settings(rag_enabled=True, rag_embedding_provider="mock")
    store = ChromaVectorStore(collection_name="test_agent_rag_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    rag_service = RAGService(vector_store=store, embedding_provider=embedder, settings=settings)

    await rag_service.ingest_text(
        content="Antigravity v5 introduces hybrid retrieval with RRF and local Chroma storage.",
        filename="antigravity_release_notes.txt",
    )

    # 2. Setup ToolRegistry with KnowledgeSearchTool
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(rag_service=rag_service))

    # 3. Setup MockLLMProvider with tool call followed by final answer
    mock_llm = MockLLMProvider(
        responses=[
            # Step 1: Agent decides to search knowledge base
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_rag_1",
                        name="knowledge_search",
                        arguments={"query": "Antigravity v5 hybrid retrieval features"},
                    )
                ],
                model="mock-llm",
            ),
            # Step 2: Agent synthesizes final answer using tool result
            LLMResponse(
                content="Antigravity v5 features hybrid retrieval combining RRF and local Chroma.",
                tool_calls=[],
                model="mock-llm",
                finish_reason="stop",
            ),
        ]
    )

    agent = ToolCallingAgent(
        provider=mock_llm,
        registry=registry,
        max_iterations=3,
    )

    # 4. Run agent
    response = await agent.run("What are the hybrid retrieval features in Antigravity v5?")

    assert response.status == "completed"
    assert "hybrid retrieval" in response.answer
    assert "RRF" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "knowledge_search"
    assert response.tool_calls[0].success is True
    assert response.tool_calls[0].result.get("count") >= 1


@pytest.mark.asyncio
async def test_agent_with_default_tool_registry_rag_enabled():
    """Verify build_default_tool_registry includes knowledge_search and is usable by agent."""
    settings = Settings(rag_enabled=True, rag_embedding_provider="mock")
    store = ChromaVectorStore(collection_name="test_default_reg_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    rag_service = RAGService(vector_store=store, embedding_provider=embedder, settings=settings)

    await rag_service.ingest_text(
        content="Secret project codename is Project Chronos.",
        filename="classified.txt",
    )

    registry = build_default_tool_registry(settings=settings, rag_service=rag_service)
    assert registry.has_tool("knowledge_search")
    assert registry.has_tool("calculator")
    assert registry.has_tool("http_get")

    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_kb_1",
                        name="knowledge_search",
                        arguments={"query": "secret project codename"},
                    )
                ],
            ),
            LLMResponse(
                content="The secret project codename is Project Chronos.",
                tool_calls=[],
            ),
        ]
    )

    agent = ToolCallingAgent(provider=mock_llm, registry=registry)
    result = await agent.run("What is the secret project codename?")

    assert result.status == "completed"
    assert "Project Chronos" in result.answer
