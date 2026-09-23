"""RAG Service coordinating document ingestion, hybrid retrieval, and grounded generation."""

from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from fastapi import Depends

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.llm.factory import get_llm_provider
from src.app.rag.chunking import RecursiveChunker
from src.app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from src.app.rag.hybrid_retriever import HybridRetriever
from src.app.rag.ingestion import DocumentLoader
from src.app.rag.models import Citation, Document, DocumentChunk, RetrievalResult
from src.app.rag.vector_store import VectorStore

logger = get_logger(__name__)


class RAGService:
    """Core RAG orchestration service."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        retriever: Optional[HybridRetriever] = None,
        loader: Optional[DocumentLoader] = None,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider or get_embedding_provider(
            settings=self.settings
        )
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            alpha=self.settings.rag_hybrid_alpha,
        )
        self.loader = loader or DocumentLoader(
            chunker=RecursiveChunker(
                chunk_size=self.settings.rag_chunk_size,
                chunk_overlap=self.settings.rag_chunk_overlap,
            ),
            max_file_size_bytes=self.settings.rag_max_file_size_bytes,
        )
        self.llm_provider = llm_provider

    async def ingest_text(
        self,
        content: str,
        filename: str = "document.txt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Ingest raw text content, generate embeddings, and index into the vector store."""
        document, chunks = self.loader.load_from_text(
            content=content,
            filename=filename,
            metadata=metadata,
        )
        await self._index_chunks(chunks)
        return document

    async def ingest_file(
        self,
        file_path_or_buffer: Union[str, Path, BinaryIO],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Ingest file (txt, md, pdf), generate embeddings, and index into the vector store."""
        document, chunks = self.loader.load_from_file(
            file_path_or_buffer=file_path_or_buffer,
            filename=filename,
            metadata=metadata,
        )
        await self._index_chunks(chunks)
        return document

    async def _index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed and store chunks in the vector store."""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed_documents(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        await self.vector_store.add_chunks(chunks)
        logger.info(f"Successfully indexed {len(chunks)} chunks into VectorStore")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve most relevant chunks for a search query."""
        k = top_k or self.settings.rag_default_top_k
        return await self.retriever.retrieve(
            query=query,
            top_k=k,
            filter_metadata=filter_metadata,
            strategy=strategy,
        )

    async def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full RAG generation pipeline: retrieval -> context -> answer + citations."""
        results = await self.retrieve(
            query=query_text,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        citations: List[Citation] = []
        context_blocks: List[str] = []

        for idx, res in enumerate(results, start=1):
            source = str(res.metadata.get("source", res.metadata.get("filename", "unknown")))
            page = res.metadata.get("page_number")
            snippet = res.content[:200] + ("..." if len(res.content) > 200 else "")

            citations.append(
                Citation(
                    document_id=res.document_id,
                    source=source,
                    page_number=page,
                    chunk_id=res.chunk_id,
                    score=res.score,
                    snippet=snippet,
                )
            )

            header = f"[Source {idx}: {source}" + (f", Page {page}" if page else "") + "]"
            context_blocks.append(f"{header}\n{res.content}")

        context_str = (
            "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."
        )

        default_system = (
            "You are a factual knowledge assistant. Use ONLY the provided context to answer. "
            "If the context does not contain enough information to answer the question, "
            "state clearly that you do not know based on the provided documents. Always cite."
        )

        user_content = f"Context:\n{context_str}\n\nQuestion: {query_text}"

        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": user_content},
        ]

        llm = self.llm_provider or get_llm_provider(self.settings)
        response = await llm.generate(messages=messages)

        return {
            "query": query_text,
            "answer": response.content,
            "citations": [c.model_dump() for c in citations],
            "retrieved_chunks": [r.model_dump() for r in results],
            "model": response.model,
            "usage": getattr(response, "usage", None),
        }

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return await self.vector_store.delete_document(document_id)

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List indexed documents."""
        return await self.vector_store.list_documents()

    async def get_stats(self) -> Dict[str, Any]:
        """Return knowledge base statistics."""
        count = await self.vector_store.count()
        docs = await self.list_documents()
        return {
            "total_chunks": count,
            "total_documents": len(docs),
            "embedding_provider": type(self.embedding_provider).__name__,
            "embedding_dimension": self.embedding_provider.dimension,
            "vector_store": type(self.vector_store).__name__,
        }


# Global singleton instance for application runtime
_rag_service_instance: Optional[RAGService] = None


def get_rag_service(
    settings: Optional[Settings] = Depends(get_settings),
) -> RAGService:
    """Factory providing singleton RAGService for the runtime application."""
    global _rag_service_instance
    if _rag_service_instance is None:
        app_settings = settings or get_settings()
        from src.app.rag.chroma_store import ChromaVectorStore

        vector_store = ChromaVectorStore(
            collection_name=app_settings.rag_collection_name,
            persist_directory=app_settings.rag_persist_directory,
        )
        _rag_service_instance = RAGService(
            vector_store=vector_store,
            settings=app_settings,
        )
    return _rag_service_instance


def reset_rag_service() -> None:
    """Reset the singleton instance (primarily for testing)."""
    global _rag_service_instance
    _rag_service_instance = None
