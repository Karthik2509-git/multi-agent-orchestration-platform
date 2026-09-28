"""ChromaDB implementation of the MemoryStore interface with strict scope isolation."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.memory.interfaces import MemoryStore
from src.app.memory.models import MemoryRecord, MemorySearchResult, MemoryType
from src.app.rag.embeddings import EmbeddingProvider, get_embedding_provider

logger = get_logger(__name__)


class ChromaMemoryStore(MemoryStore):
    """Long-term semantic memory store backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: Optional[str] = "./data/chroma_memory",
        embedding_provider: Optional[EmbeddingProvider] = None,
        ephemeral: bool = False,
        settings: Optional[Settings] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.ephemeral = ephemeral
        app_settings = settings or get_settings()
        self.embedding_provider = embedding_provider or get_embedding_provider(
            settings=app_settings
        )
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize ChromaDB client and memory collection."""
        try:
            if self.ephemeral or not self.persist_directory:
                logger.info("Initializing in-memory Chroma EphemeralClient for MemoryStore.")
                self._client = chromadb.EphemeralClient(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            else:
                persist_path = Path(self.persist_directory).resolve()
                persist_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Initializing Chroma PersistentClient for MemoryStore at '%s'", persist_path
                )
                self._client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error("Failed to initialize Chroma client for MemoryStore: %s", e)
            raise RuntimeError(f"Chroma memory initialization failed: {e}") from e

    def _serialize_metadata(self, record: MemoryRecord) -> Dict[str, Any]:
        """Convert MemoryRecord fields to flat Chroma-compatible metadata dictionary."""
        sanitized: Dict[str, Any] = {
            "memory_id": record.id,
            "scope_id": record.scope_id,
            "memory_type": record.memory_type.value,
            "importance": float(record.importance),
            "access_count": int(record.access_count),
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat(),
            "source_task_ids_json": json.dumps(record.source_task_ids),
        }
        if record.expires_at:
            sanitized["expires_at"] = record.expires_at.isoformat()
        if record.metadata:
            sanitized["extra_metadata_json"] = json.dumps(record.metadata)
        return sanitized

    def _deserialize_record(self, doc_id: str, content: str, meta: Dict[str, Any]) -> MemoryRecord:
        """Construct MemoryRecord from Chroma metadata and document text."""
        source_task_ids = []
        if "source_task_ids_json" in meta:
            try:
                source_task_ids = json.loads(meta["source_task_ids_json"])
            except Exception:
                source_task_ids = []

        extra_meta = {}
        if "extra_metadata_json" in meta:
            try:
                extra_meta = json.loads(meta["extra_metadata_json"])
            except Exception:
                extra_meta = {}

        expires_at = None
        if "expires_at" in meta and meta["expires_at"]:
            try:
                expires_at = datetime.fromisoformat(meta["expires_at"])
            except Exception:
                expires_at = None

        return MemoryRecord(
            id=meta.get("memory_id", doc_id),
            scope_id=meta.get("scope_id", "default"),
            content=content,
            memory_type=MemoryType(meta.get("memory_type", MemoryType.DOMAIN_FACT.value)),
            importance=float(meta.get("importance", 0.5)),
            access_count=int(meta.get("access_count", 0)),
            created_at=datetime.fromisoformat(
                meta.get("created_at", datetime.now(timezone.utc).isoformat())
            ),
            updated_at=datetime.fromisoformat(
                meta.get("updated_at", datetime.now(timezone.utc).isoformat())
            ),
            last_accessed_at=datetime.fromisoformat(
                meta.get("last_accessed_at", datetime.now(timezone.utc).isoformat())
            ),
            expires_at=expires_at,
            source_task_ids=source_task_ids,
            metadata=extra_meta,
        )

    async def add_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Upsert a memory record into the Chroma collection."""
        if not record.embedding:
            record.embedding = await self.embedding_provider.embed_query(record.content)

        metadata = self._serialize_metadata(record)

        await asyncio.to_thread(
            self._collection.upsert,
            ids=[record.id],
            documents=[record.content],
            metadatas=[metadata],
            embeddings=[record.embedding],
        )
        return record

    async def get_memory(
        self, memory_id: str, scope_id: Optional[str] = None
    ) -> Optional[MemoryRecord]:
        """Retrieve a specific memory by ID, enforcing scope validation if provided."""
        where_filter = None
        if scope_id:
            where_filter = {"scope_id": scope_id}

        result = await asyncio.to_thread(
            self._collection.get,
            ids=[memory_id],
            where=where_filter,
            include=["documents", "metadatas"],
        )

        if not result["ids"] or len(result["ids"]) == 0:
            return None

        content = result["documents"][0]
        meta = result["metadatas"][0] if result.get("metadatas") else {}
        return self._deserialize_record(memory_id, content, meta)

    async def search_memories(
        self,
        query: str,
        scope_id: str = "default",
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        """Search memories with strict scope isolation."""
        count = await self.count(scope_id=scope_id)
        if count == 0:
            return []

        query_embedding = await self.embedding_provider.embed_query(query)

        where_clauses: List[Dict[str, Any]] = [{"scope_id": scope_id}]
        if memory_type:
            where_clauses.append({"memory_type": memory_type.value})
        if filter_metadata:
            for k, v in filter_metadata.items():
                where_clauses.append({k: v})

        where_filter: Dict[str, Any] = (
            {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]
        )

        n_results = min(top_k, count)
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output: List[MemorySearchResult] = []
        if not results["ids"] or len(results["ids"][0]) == 0:
            return output

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, mid in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0

            # Convert cosine distance to cosine similarity: sim = 1.0 - dist
            similarity = max(0.0, min(1.0, 1.0 - dist))

            record = self._deserialize_record(mid, content, meta)
            # Skip expired records
            if record.is_expired:
                continue

            output.append(
                MemorySearchResult(memory=record, score=similarity, similarity=similarity)
            )

        return output

    async def list_memories(
        self,
        scope_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List memories with optional scope and type filters."""
        where_clauses: List[Dict[str, Any]] = []
        if scope_id:
            where_clauses.append({"scope_id": scope_id})
        if memory_type:
            where_clauses.append({"memory_type": memory_type.value})

        where_filter = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}

        results = await asyncio.to_thread(
            self._collection.get,
            where=where_filter,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )

        records: List[MemoryRecord] = []
        if not results["ids"]:
            return records

        for i, mid in enumerate(results["ids"]):
            content = results["documents"][i] if results.get("documents") else ""
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            records.append(self._deserialize_record(mid, content, meta))

        return records

    async def update_memory(self, record: MemoryRecord) -> Optional[MemoryRecord]:
        """Update an existing memory record."""
        record.updated_at = datetime.now(timezone.utc)
        return await self.add_memory(record)

    async def delete_memory(self, memory_id: str, scope_id: Optional[str] = None) -> bool:
        """Delete a memory record by ID with optional scope verification."""
        existing = await self.get_memory(memory_id, scope_id=scope_id)
        if not existing:
            return False

        await asyncio.to_thread(
            self._collection.delete,
            ids=[memory_id],
        )
        return True

    async def count(self, scope_id: Optional[str] = None) -> int:
        """Return count of memories, optionally scoped."""
        if not scope_id:
            return await asyncio.to_thread(self._collection.count)

        res = await asyncio.to_thread(
            self._collection.get,
            where={"scope_id": scope_id},
        )
        return len(res["ids"]) if res.get("ids") else 0
