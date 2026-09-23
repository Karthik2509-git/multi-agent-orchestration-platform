"""Abstract VectorStore interface for RAG vector databases."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.app.rag.models import DocumentChunk, RetrievalResult


class VectorStore(ABC):
    """Abstract interface defining required vector store operations."""

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add or update document chunks with their embeddings in the vector store."""
        pass

    @abstractmethod
    async def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform dense semantic similarity search against indexed vector chunks."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID. Returns count of deleted chunks."""
        pass

    @abstractmethod
    async def get_all_chunks(
        self, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Retrieve all indexed chunks, optionally filtered by metadata."""
        pass

    @abstractmethod
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct ingested documents with metadata summary."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return total number of chunks currently stored."""
        pass
