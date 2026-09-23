"""RAG (Retrieval-Augmented Generation) & Knowledge subsystem."""

from src.app.rag.chunking import RecursiveChunker
from src.app.rag.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from src.app.rag.hybrid_retriever import BM25Scorer, HybridRetriever
from src.app.rag.ingestion import DocumentLoader
from src.app.rag.models import Citation, Document, DocumentChunk, RetrievalResult
from src.app.rag.service import RAGService, get_rag_service
from src.app.rag.vector_store import VectorStore

__all__ = [
    "Citation",
    "Document",
    "DocumentChunk",
    "RetrievalResult",
    "RecursiveChunker",
    "DocumentLoader",
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
    "VectorStore",
    "BM25Scorer",
    "HybridRetriever",
    "RAGService",
    "get_rag_service",
]
