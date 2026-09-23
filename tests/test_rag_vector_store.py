"""Tests for ChromaVectorStore implementation."""

import shutil
import tempfile

import pytest

from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.models import DocumentChunk


@pytest.mark.asyncio
async def test_add_and_search_vector():
    """Test adding chunks and performing vector search."""
    store = ChromaVectorStore(collection_name="test_search_kb", ephemeral=True)
    chunks = [
        DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            content="FastAPI is a modern, fast web framework for building APIs with Python.",
            index=0,
            metadata={"source": "fastapi.md", "topic": "web"},
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
        DocumentChunk(
            id="chunk-2",
            document_id="doc-2",
            content="LangGraph builds stateful multi-actor agent apps.",
            index=0,
            metadata={"source": "langgraph.md", "topic": "agents"},
            embedding=[0.0, 0.0, 0.8, 0.2],
        ),
    ]

    await store.add_chunks(chunks)
    assert await store.count() == 2

    results = await store.search_vector(
        query_embedding=[0.85, 0.15, 0.0, 0.0],
        top_k=2,
    )
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score > 0.8


@pytest.mark.asyncio
async def test_filter_metadata():
    """Test searching with metadata filter."""
    store = ChromaVectorStore(collection_name="test_filter_kb", ephemeral=True)
    chunks = [
        DocumentChunk(
            id="chunk-a",
            document_id="doc-a",
            content="Alpha content",
            index=0,
            metadata={"category": "finance", "year": 2024},
            embedding=[0.5, 0.5],
        ),
        DocumentChunk(
            id="chunk-b",
            document_id="doc-b",
            content="Beta content",
            index=0,
            metadata={"category": "engineering", "year": 2024},
            embedding=[0.5, 0.5],
        ),
    ]
    await store.add_chunks(chunks)

    results = await store.search_vector(
        query_embedding=[0.5, 0.5],
        top_k=5,
        filter_metadata={"category": "engineering"},
    )
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-b"


@pytest.mark.asyncio
async def test_delete_document():
    """Test deleting document removes all corresponding chunks."""
    store = ChromaVectorStore(collection_name="test_delete_kb", ephemeral=True)
    chunks = [
        DocumentChunk(
            id="chunk-x1",
            document_id="doc-del",
            content="Delete me part 1",
            index=0,
            embedding=[0.1, 0.2],
        ),
        DocumentChunk(
            id="chunk-x2",
            document_id="doc-del",
            content="Delete me part 2",
            index=1,
            embedding=[0.1, 0.2],
        ),
        DocumentChunk(
            id="chunk-keep",
            document_id="doc-keep",
            content="Keep me",
            index=0,
            embedding=[0.9, 0.1],
        ),
    ]
    await store.add_chunks(chunks)
    assert await store.count() == 3

    deleted = await store.delete_document("doc-del")
    assert deleted == 2
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_persistent_chroma_storage():
    """Test persistent ChromaVectorStore preserves data across restarts."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Instance 1: write data
        store1 = ChromaVectorStore(
            collection_name="test_persist_kb",
            persist_directory=temp_dir,
            ephemeral=False,
        )
        chunk = DocumentChunk(
            id="persistent-chunk-1",
            document_id="doc-persisted",
            content="Data should survive store re-instantiation.",
            index=0,
            metadata={"source": "persist.txt"},
            embedding=[0.7, 0.7],
        )
        await store1.add_chunks([chunk])
        assert await store1.count() == 1

        # Instance 2: re-open from same directory
        store2 = ChromaVectorStore(
            collection_name="test_persist_kb",
            persist_directory=temp_dir,
            ephemeral=False,
        )
        assert await store2.count() == 1
        docs = await store2.list_documents()
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc-persisted"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
