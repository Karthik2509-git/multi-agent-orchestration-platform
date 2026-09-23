"""Tests for RAG domain models."""

from src.app.rag.models import Citation, Document, DocumentChunk, RetrievalResult


def test_document_model_instantiation():
    """Test Document model creation and default values."""
    doc = Document(filename="readme.md", content="# Hello World")
    assert doc.id is not None
    assert doc.filename == "readme.md"
    assert doc.content == "# Hello World"
    assert doc.created_at is not None
    assert doc.metadata == {}


def test_document_chunk_model():
    """Test DocumentChunk fields and embedding storage."""
    chunk = DocumentChunk(
        document_id="doc-123",
        content="This is chunk content",
        index=0,
        metadata={"source": "test.txt", "page_number": 1},
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.document_id == "doc-123"
    assert chunk.index == 0
    assert chunk.embedding == [0.1, 0.2, 0.3]
    assert chunk.metadata["page_number"] == 1


def test_retrieval_result_model():
    """Test RetrievalResult model fields."""
    res = RetrievalResult(
        chunk_id="chk-1",
        document_id="doc-1",
        content="Some matched content",
        score=0.88,
        source_type="hybrid_rrf",
    )
    assert res.chunk_id == "chk-1"
    assert res.score == 0.88
    assert res.source_type == "hybrid_rrf"


def test_citation_model():
    """Test Citation model fields and snippet serialization."""
    citation = Citation(
        document_id="doc-1",
        source="guide.pdf",
        page_number=3,
        chunk_id="chunk-45",
        score=0.92,
        snippet="Relevant excerpt from page 3",
    )
    assert citation.source == "guide.pdf"
    assert citation.page_number == 3
    assert citation.score == 0.92
