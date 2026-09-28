"""High-level MemoryService orchestrator."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.memory.consolidation import MemoryConsolidator
from src.app.memory.extraction import MemoryExtractor
from src.app.memory.interfaces import MemoryStore
from src.app.memory.models import MemoryRecord, MemorySearchResult, MemoryStats, MemoryType
from src.app.memory.retrieval import MemoryRetriever
from src.app.memory.stores.chroma_memory_store import ChromaMemoryStore

logger = get_logger(__name__)


class MemoryService:
    """Unified service coordinating long-term memory operations."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        retriever: Optional[MemoryRetriever] = None,
        extractor: Optional[MemoryExtractor] = None,
        consolidator: Optional[MemoryConsolidator] = None,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or ChromaMemoryStore(
            collection_name=self.settings.memory_collection_name,
            persist_directory=self.settings.memory_persist_directory,
            settings=self.settings,
        )
        self.retriever = retriever or MemoryRetriever(
            store=self.store,
            similarity_threshold=self.settings.memory_similarity_threshold,
        )
        self.extractor = extractor or MemoryExtractor(llm_provider=llm_provider)
        self.consolidator = consolidator or MemoryConsolidator(
            store=self.store,
            similarity_threshold=self.settings.memory_consolidation_threshold,
        )

    async def add_memory(
        self,
        content: str,
        scope_id: str = "default",
        memory_type: MemoryType = MemoryType.DOMAIN_FACT,
        importance: float = 0.5,
        expires_at: Optional[datetime] = None,
        source_task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Create and store a new memory record."""
        source_task_ids = [source_task_id] if source_task_id else []
        record = MemoryRecord(
            scope_id=scope_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            expires_at=expires_at,
            source_task_ids=source_task_ids,
            metadata=metadata or {},
        )
        return await self.store.add_memory(record)

    async def get_memory(
        self, memory_id: str, scope_id: Optional[str] = None
    ) -> Optional[MemoryRecord]:
        """Retrieve a specific memory with optional scope verification."""
        return await self.store.get_memory(memory_id, scope_id=scope_id)

    async def search_memories(
        self,
        query: str,
        scope_id: str = "default",
        top_k: Optional[int] = None,
        memory_type: Optional[MemoryType] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        """Search and rank memories using the composite ranking formula."""
        k = top_k or self.settings.memory_default_top_k
        return await self.retriever.retrieve(
            query=query,
            scope_id=scope_id,
            top_k=k,
            memory_type=memory_type,
            filter_metadata=filter_metadata,
        )

    async def list_memories(
        self,
        scope_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List stored memories."""
        return await self.store.list_memories(
            scope_id=scope_id,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        )

    async def delete_memory(self, memory_id: str, scope_id: Optional[str] = None) -> bool:
        """Delete a memory record."""
        return await self.store.delete_memory(memory_id, scope_id=scope_id)

    async def consolidate(self, scope_id: str = "default") -> Dict[str, Any]:
        """Consolidate duplicate memories and remove expired items."""
        return await self.consolidator.consolidate_scope(scope_id=scope_id)

    async def extract_and_store_memories(
        self,
        task: str,
        final_answer: str,
        agent_results: Optional[Dict[str, str]] = None,
        scope_id: str = "default",
        task_id: Optional[str] = None,
        use_llm: bool = False,
    ) -> List[MemoryRecord]:
        """Extract post-task memories and persist them to long-term storage."""
        candidates = await self.extractor.extract(
            task=task,
            final_answer=final_answer,
            agent_results=agent_results,
            scope_id=scope_id,
            task_id=task_id,
            use_llm=use_llm,
        )

        stored: List[MemoryRecord] = []
        for cand in candidates:
            rec = await self.store.add_memory(cand)
            stored.append(rec)

        return stored

    async def get_stats(self, scope_id: Optional[str] = None) -> MemoryStats:
        """Compute aggregated statistics for stored memories."""
        records = await self.store.list_memories(scope_id=scope_id, limit=1000)
        by_type: Dict[str, int] = {}
        scope_counts: Dict[str, int] = {}

        for r in records:
            t = r.memory_type.value
            by_type[t] = by_type.get(t, 0) + 1
            s = r.scope_id
            scope_counts[s] = scope_counts.get(s, 0) + 1

        return MemoryStats(
            total_memories=len(records),
            memories_by_type=by_type,
            scope_counts=scope_counts,
            storage_backend=type(self.store).__name__,
        )


_global_memory_service: Optional[MemoryService] = None


def get_memory_service(settings: Optional[Settings] = Depends(get_settings)) -> MemoryService:
    """Dependency provider returning singleton MemoryService."""
    global _global_memory_service
    if _global_memory_service is None:
        app_settings = settings or get_settings()
        _global_memory_service = MemoryService(settings=app_settings)
    return _global_memory_service


def reset_memory_service() -> None:
    """Reset singleton memory service instance."""
    global _global_memory_service
    _global_memory_service = None
