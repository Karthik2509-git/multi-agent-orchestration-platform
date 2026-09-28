"""Unit tests for MemoryConsolidator."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.app.memory.consolidation import MemoryConsolidator
from src.app.memory.models import MemoryRecord
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore
from src.app.rag.embeddings import MockEmbeddingProvider


@pytest.fixture
def mock_consolidator():
    embedder = MockEmbeddingProvider(dimension=16)
    store = ChromaMemoryStore(
        collection_name=f"test_consolidation_mem_{uuid4().hex[:8]}",
        embedding_provider=embedder,
        ephemeral=True,
    )
    consolidator = MemoryConsolidator(store=store, similarity_threshold=0.85)
    return store, consolidator


@pytest.mark.asyncio
async def test_cleanup_expired_memories(mock_consolidator):
    """Verify expired memories are purged from storage."""
    store, consolidator = mock_consolidator
    now = datetime.now(timezone.utc)

    # 1 valid record, 1 expired record
    valid = MemoryRecord(content="Valid permanent memory", expires_at=now + timedelta(days=1))
    expired = MemoryRecord(content="Old expired memory", expires_at=now - timedelta(days=1))

    await store.add_memory(valid)
    await store.add_memory(expired)
    assert await store.count() == 2

    cleaned = await consolidator.cleanup_expired(scope_id="default")
    assert cleaned == 1
    assert await store.count() == 1
    assert await store.get_memory(valid.id) is not None
    assert await store.get_memory(expired.id) is None


@pytest.mark.asyncio
async def test_safe_duplicate_consolidation(mock_consolidator):
    """Verify duplicate records are merged without losing metadata, provenance, or access count."""
    store, consolidator = mock_consolidator

    # Same text content -> MockEmbeddingProvider produces identical embeddings (similarity = 1.0)
    rec1 = MemoryRecord(
        content="Always verify calculations with CalculatorTool.",
        importance=0.7,
        access_count=3,
        source_task_ids=["task_1"],
        metadata={"author": "karthik"},
    )
    rec2 = MemoryRecord(
        content="Always verify calculations with CalculatorTool.",
        importance=0.9,
        access_count=5,
        source_task_ids=["task_2"],
        metadata={"verified": True},
    )

    await store.add_memory(rec1)
    await store.add_memory(rec2)
    assert await store.count() == 2

    res = await consolidator.consolidate_scope(scope_id="default")
    assert res["merged_count"] == 1
    assert res["remaining_count"] == 1

    # Remaining record should have merged task IDs, summed access count,
    # max importance, and combined metadata
    remaining = await store.list_memories(scope_id="default")
    assert len(remaining) == 1
    m = remaining[0]
    assert m.access_count == 8  # 3 + 5
    assert m.importance == 0.9  # max(0.7, 0.9)
    assert set(m.source_task_ids) == {"task_1", "task_2"}
    assert m.metadata.get("author") == "karthik"
    assert m.metadata.get("verified") is True
    assert "consolidated_at" in m.metadata
