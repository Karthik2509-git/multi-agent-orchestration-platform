"""Tests for RecursiveChunker."""

import pytest

from src.app.rag.chunking import RecursiveChunker


def test_chunker_initialization():
    """Test valid and invalid chunker configurations."""
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        RecursiveChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RecursiveChunker(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        RecursiveChunker(chunk_size=500, chunk_overlap=500)


def test_chunk_empty_document():
    """Test chunking empty or whitespace text returns empty list."""
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    assert chunker.chunk_document("doc-1", "") == []
    assert chunker.chunk_document("doc-1", "   \n\n  ") == []


def test_chunk_short_document():
    """Test chunking text shorter than chunk_size returns single chunk."""
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
    text = "This is a short paragraph that easily fits within one chunk."
    chunks = chunker.chunk_document("doc-1", text, base_metadata={"source": "test.txt"})

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].index == 0
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].metadata["source"] == "test.txt"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_long_document_with_paragraphs():
    """Test chunking text with paragraphs splits logically and applies overlap."""
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    p1 = "Paragraph 1: Multi-agent systems involve coordinating several autonomous AI entities."
    p2 = "Paragraph 2: Each agent focuses on a specific specialty like research, code, or math."
    p3 = "Paragraph 3: A supervisor orchestrator directs the flow of execution and aggregates."
    text = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = chunker.chunk_document("doc-1", text)
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert chunk.document_id == "doc-1"
        assert len(chunk.content) > 0
