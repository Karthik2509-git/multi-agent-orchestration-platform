"""Unit tests for ChromaMemoryStore CRUD, search, and scope isolation."""

import pytest

from src.app.memory.models import MemoryRecord, MemoryType
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore
from src.app.rag.embeddings import MockEmbeddingProvider


@pytest.fixture
def mock_memory_store():
    """Provide an in-memory ephemeral ChromaMemoryStore with deterministic MockEmbeddingProvider."""
    embedder = MockEmbeddingProvider(dimension=16)
    store = ChromaMemoryStore(
        collection_name="test_agent_memory",
        embedding_provider=embedder,
        ephemeral=True,
    )
    return store


@pytest.mark.asyncio
async def test_memory_store_crud(mock_memory_store):
    """Test creating, reading, updating, and deleting memory records."""
    rec = MemoryRecord(
        content="Always format financial outputs with 2 decimal places.",
        scope_id="user_alpha",
        memory_type=MemoryType.USER_PREFERENCE,
        importance=0.8,
    )

    # 1. Add
    added = await mock_memory_store.add_memory(rec)
    assert added.id == rec.id
    assert await mock_memory_store.count(scope_id="user_alpha") == 1

    # 2. Get
    fetched = await mock_memory_store.get_memory(rec.id, scope_id="user_alpha")
    assert fetched is not None
    assert fetched.content == rec.content
    assert fetched.memory_type == MemoryType.USER_PREFERENCE
    assert fetched.importance == 0.8

    # 3. Update
    fetched.importance = 0.95
    fetched.content = "Always format financial outputs with 2 decimal places and currency symbols."
    updated = await mock_memory_store.update_memory(fetched)
    assert updated.importance == 0.95

    # 4. Delete
    deleted = await mock_memory_store.delete_memory(rec.id, scope_id="user_alpha")
    assert deleted is True
    assert await mock_memory_store.get_memory(rec.id, scope_id="user_alpha") is None


@pytest.mark.asyncio
async def test_memory_scope_isolation(mock_memory_store):
    """Verify strict scope isolation: Scope A memories are never returned in Scope B search/list."""
    rec_a = MemoryRecord(
        content="Alpha user prefers detailed explanations.",
        scope_id="tenant_a",
        memory_type=MemoryType.USER_PREFERENCE,
    )
    rec_b = MemoryRecord(
        content="Beta user prefers brief summaries.",
        scope_id="tenant_b",
        memory_type=MemoryType.USER_PREFERENCE,
    )

    await mock_memory_store.add_memory(rec_a)
    await mock_memory_store.add_memory(rec_b)

    # Search in tenant_a
    results_a = await mock_memory_store.search_memories(
        query="user preference",
        scope_id="tenant_a",
    )
    assert len(results_a) == 1
    assert results_a[0].memory.scope_id == "tenant_a"
    assert results_a[0].memory.content == rec_a.content

    # Search in tenant_b
    results_b = await mock_memory_store.search_memories(
        query="user preference",
        scope_id="tenant_b",
    )
    assert len(results_b) == 1
    assert results_b[0].memory.scope_id == "tenant_b"
    assert results_b[0].memory.content == rec_b.content

    # Cross-scope get should return None
    cross_get = await mock_memory_store.get_memory(rec_a.id, scope_id="tenant_b")
    assert cross_get is None


@pytest.mark.asyncio
async def test_cross_scope_delete_prevention(mock_memory_store):
    """Verify that a delete request specifying Scope B cannot delete a memory in Scope A."""
    rec_a = MemoryRecord(
        content="Confidential strategy for company A",
        scope_id="company_a",
        memory_type=MemoryType.DOMAIN_FACT,
    )
    await mock_memory_store.add_memory(rec_a)

    # Attempt to delete with scope company_b
    deleted = await mock_memory_store.delete_memory(rec_a.id, scope_id="company_b")
    assert deleted is False

    # Verify memory A is intact
    still_there = await mock_memory_store.get_memory(rec_a.id, scope_id="company_a")
    assert still_there is not None
    assert still_there.content == rec_a.content


@pytest.mark.asyncio
async def test_cross_scope_list_and_count_isolation(mock_memory_store):
    """Verify list_memories and count strictly isolate records by scope_id."""
    for i in range(3):
        await mock_memory_store.add_memory(
            MemoryRecord(
                content=f"Company X fact {i}",
                scope_id="scope_x",
                memory_type=MemoryType.DOMAIN_FACT,
            )
        )
    for i in range(2):
        await mock_memory_store.add_memory(
            MemoryRecord(
                content=f"Company Y fact {i}",
                scope_id="scope_y",
                memory_type=MemoryType.DOMAIN_FACT,
            )
        )

    # List scope_x
    list_x = await mock_memory_store.list_memories(scope_id="scope_x")
    assert len(list_x) == 3
    assert all(r.scope_id == "scope_x" for r in list_x)

    # List scope_y
    list_y = await mock_memory_store.list_memories(scope_id="scope_y")
    assert len(list_y) == 2
    assert all(r.scope_id == "scope_y" for r in list_y)

    # Count
    assert await mock_memory_store.count(scope_id="scope_x") == 3
    assert await mock_memory_store.count(scope_id="scope_y") == 2
