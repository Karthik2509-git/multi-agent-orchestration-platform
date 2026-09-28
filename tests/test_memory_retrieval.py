"""Unit tests for MemoryRetriever ranking formula and context formatting."""

from datetime import datetime, timedelta, timezone

import pytest

from src.app.memory.models import MemoryRecord, MemoryType
from src.app.memory.retrieval import MemoryRetriever
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore
from src.app.rag.embeddings import MockEmbeddingProvider


@pytest.fixture
def mock_retriever():
    embedder = MockEmbeddingProvider(dimension=16)
    store = ChromaMemoryStore(
        collection_name="test_retrieval_mem",
        embedding_provider=embedder,
        ephemeral=True,
    )
    retriever = MemoryRetriever(
        store=store,
        similarity_weight=0.65,
        importance_weight=0.25,
        recency_weight=0.10,
        similarity_threshold=0.0,
    )
    return store, retriever


@pytest.mark.asyncio
async def test_composite_ranking_formula(mock_retriever):
    """Verify formula score = 0.65*sim + 0.25*imp + 0.10*recency."""
    store, retriever = mock_retriever
    now = datetime.now(timezone.utc)

    # Fresh record with high importance
    rec_high = MemoryRecord(
        content="Important recent fact about database replication.",
        scope_id="default",
        importance=1.0,
        updated_at=now,
    )
    # Stale record with low importance
    rec_low = MemoryRecord(
        content="Old low importance note.",
        scope_id="default",
        importance=0.2,
        updated_at=now - timedelta(days=60),
    )

    score_high = retriever.compute_composite_score(similarity=0.9, record=rec_high, now=now)
    score_low = retriever.compute_composite_score(similarity=0.9, record=rec_low, now=now)

    # score_high: 0.65*(0.9) + 0.25*(1.0) + 0.10*(1.0) = 0.585 + 0.25 + 0.10 = 0.935
    assert score_high >= 0.90
    assert score_high > score_low


@pytest.mark.asyncio
async def test_planning_context_formatting(mock_retriever):
    """Verify formatting of retrieved memories for supervisor planning context."""
    store, retriever = mock_retriever
    rec = MemoryRecord(
        content="Always verify calculations using the CalculatorTool.",
        memory_type=MemoryType.TASK_LESSON,
        importance=0.85,
    )
    await store.add_memory(rec)

    results = await retriever.retrieve(
        query="CalculatorTool arithmetic",
        scope_id="default",
        top_k=1,
    )

    formatted = retriever.format_for_planning(results)
    assert "Relevant Memories & Past Context:" in formatted
    assert "[Task Lesson]" in formatted
    assert "Always verify calculations" in formatted
    assert "0.85" in formatted
