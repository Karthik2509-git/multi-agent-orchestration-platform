"""ChromaDB implementation of the VectorStore interface."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.app.core.logging import get_logger
from src.app.rag.models import DocumentChunk, RetrievalResult
from src.app.rag.vector_store import VectorStore

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    """VectorStore backed by ChromaDB with support for persistent and ephemeral storage."""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: Optional[str] = "./data/chroma",
        ephemeral: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.ephemeral = ephemeral
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            if self.ephemeral or not self.persist_directory:
                logger.info("Initializing in-memory Chroma EphemeralClient.")
                self._client = chromadb.EphemeralClient(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            else:
                persist_path = Path(self.persist_directory).resolve()
                persist_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Initializing Chroma PersistentClient at '{persist_path}'")
                self._client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise RuntimeError(f"Chroma initialization failed: {e}") from e

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Upsert document chunks into the Chroma collection."""
        if not chunks:
            return

        def _upsert():
            ids = [c.id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = []
            for c in chunks:
                sanitized_meta = {
                    "document_id": str(c.document_id),
                    "chunk_index": int(c.index),
                }
                for k, v in c.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    elif v is not None:
                        sanitized_meta[k] = str(v)
                metadatas.append(sanitized_meta)

            embeddings = (
                [c.embedding for c in chunks]
                if all(c.embedding is not None for c in chunks)
                else None
            )

            if embeddings:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            else:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

        await asyncio.to_thread(_upsert)
        logger.debug(
            f"Successfully added {len(chunks)} chunks to Chroma collection '{self.collection_name}'"
        )

    async def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform dense semantic similarity search against stored chunk vectors."""
        total_items = await self.count()
        if total_items == 0:
            return []

        n_results = min(top_k, total_items)
        where_filter = self._build_where_filter(filter_metadata)

        def _query():
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        results = await asyncio.to_thread(_query)
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        retrieval_results: List[RetrievalResult] = []
        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else 0.0
            score = max(0.0, min(1.0, 1.0 - dist))
            doc_id = str(meta.get("document_id", ""))

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=content,
                    metadata=meta,
                    score=round(score, 4),
                    source_type="semantic",
                )
            )

        return retrieval_results

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID."""

        def _delete():
            existing = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            count = len(existing["ids"]) if existing and existing.get("ids") else 0
            if count > 0:
                self._collection.delete(where={"document_id": document_id})
            return count

        deleted_count = await asyncio.to_thread(_delete)
        logger.info(f"Deleted {deleted_count} chunks for document_id '{document_id}'")
        return deleted_count

    async def get_all_chunks(
        self, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Retrieve all indexed chunks, optionally filtered by metadata."""
        where_filter = self._build_where_filter(filter_metadata)

        def _get():
            return self._collection.get(
                where=where_filter,
                include=["documents", "metadatas", "embeddings"],
            )

        data = await asyncio.to_thread(_get)
        if not data or not data.get("ids"):
            return []

        chunks: List[DocumentChunk] = []
        ids = data["ids"]
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        embeddings = data.get("embeddings")

        for i, chunk_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            doc_id = str(meta.get("document_id", ""))
            idx = int(meta.get("chunk_index", i))
            emb = embeddings[i] if embeddings is not None and i < len(embeddings) else None

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=content,
                    index=idx,
                    metadata=meta,
                    embedding=[float(x) for x in emb] if emb is not None else None,
                )
            )

        return chunks

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct ingested documents with metadata and chunk counts."""
        chunks = await self.get_all_chunks()
        doc_map: Dict[str, Dict[str, Any]] = {}

        for c in chunks:
            doc_id = c.document_id or c.metadata.get("document_id", "unknown")
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": c.metadata.get("filename", "unknown"),
                    "source": c.metadata.get("source", "unknown"),
                    "file_type": c.metadata.get("file_type", ".txt"),
                    "page_count": c.metadata.get("page_count", 1),
                    "chunk_count": 0,
                    "created_at": c.metadata.get("created_at"),
                }
            doc_map[doc_id]["chunk_count"] += 1

        return list(doc_map.values())

    async def count(self) -> int:
        """Return total chunk count in the collection."""
        return await asyncio.to_thread(self._collection.count)

    def _build_where_filter(
        self, filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Convert a metadata filter dict to Chroma where clause format."""
        if not filter_metadata:
            return None

        items = list(filter_metadata.items())
        if len(items) == 1:
            return {items[0][0]: items[0][1]}

        return {"$and": [{k: v} for k, v in items]}
