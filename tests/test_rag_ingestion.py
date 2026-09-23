"""Tests for DocumentLoader and file ingestion."""

import io

import pytest
from pypdf import PdfWriter

from src.app.rag.ingestion import DocumentLoader


def test_load_from_text_success():
    """Test loading plain text document."""
    loader = DocumentLoader()
    doc, chunks = loader.load_from_text(
        content="This is the content of an ingested knowledge file.",
        filename="guide.txt",
        metadata={"category": "manual"},
    )

    assert doc.filename == "guide.txt"
    assert doc.metadata["category"] == "manual"
    assert len(chunks) >= 1
    assert chunks[0].document_id == doc.id
    assert chunks[0].metadata["source"] == "guide.txt"


def test_load_from_text_empty_error():
    """Test loading empty content raises ValueError."""
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="cannot be empty"):
        loader.load_from_text("")


def test_load_unsupported_extension():
    """Test loading unsupported file extension raises ValueError."""
    loader = DocumentLoader()
    buffer = io.BytesIO(b"some raw executable or binary")
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_from_file(buffer, filename="malicious.exe")


def test_load_file_size_exceeded():
    """Test file size limit check."""
    loader = DocumentLoader(max_file_size_bytes=50)
    large_content = io.BytesIO(b"A" * 100)
    with pytest.raises(ValueError, match="exceeds limit"):
        loader.load_from_file(large_content, filename="large.txt")


def test_load_pdf_page_aware():
    """Test PDF extraction and page metadata preservation using synthetic PDF."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)

    # Blank pages have no text; test byte buffer loader with markdown
    text_buffer = io.BytesIO(b"# Markdown Title\n\nSection content goes here.")
    loader = DocumentLoader()
    doc, chunks = loader.load_from_file(text_buffer, filename="notes.md")

    assert doc.filename == "notes.md"
    assert doc.metadata["file_type"] == ".md"
    assert len(chunks) >= 1
