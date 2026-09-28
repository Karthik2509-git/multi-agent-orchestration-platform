"""Memory consolidation and lifecycle management engine."""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from src.app.core.logging import get_logger
from src.app.memory.interfaces import MemoryStore

logger = get_logger(__name__)


class MemoryConsolidator:
    """Manages memory lifecycle: expiration cleanup and duplicate consolidation."""

    def __init__(
        self,
        store: MemoryStore,
        similarity_threshold: float = 0.90,
    ):
        self.store = store
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    async def cleanup_expired(self, scope_id: Optional[str] = None) -> int:
        """Purge all expired memories from the store."""
        records = await self.store.list_memories(scope_id=scope_id, limit=500)
        deleted_count = 0
        for rec in records:
            if rec.is_expired:
                success = await self.store.delete_memory(rec.id, scope_id=rec.scope_id)
                if success:
                    deleted_count += 1
        return deleted_count

    async def consolidate_scope(
        self,
        scope_id: str = "default",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Consolidate duplicate memories within a scope preserving metadata and provenance."""
        cutoff = threshold or self.similarity_threshold
        expired_deleted = await self.cleanup_expired(scope_id=scope_id)

        records = await self.store.list_memories(scope_id=scope_id, limit=500)
        if len(records) < 2:
            return {
                "scope_id": scope_id,
                "expired_deleted": expired_deleted,
                "merged_count": 0,
                "remaining_count": len(records),
            }

        # Ensure embeddings are present for comparison
        for r in records:
            if not r.embedding:
                try:
                    r.embedding = await self.store.embedding_provider.embed_query(r.content)
                except Exception:
                    pass

        merged_ids: Set[str] = set()
        merged_count = 0
        now = datetime.now(timezone.utc)

        for i in range(len(records)):
            primary = records[i]
            if primary.id in merged_ids or not primary.embedding:
                continue

            for j in range(i + 1, len(records)):
                candidate = records[j]
                if candidate.id in merged_ids or not candidate.embedding:
                    continue

                sim = self._cosine_similarity(primary.embedding, candidate.embedding)
                if sim >= cutoff:
                    # Merge candidate into primary
                    primary.access_count += candidate.access_count
                    primary.importance = max(primary.importance, candidate.importance)

                    # Merge source task IDs (union list)
                    combined_tasks = list(set(primary.source_task_ids + candidate.source_task_ids))
                    primary.source_task_ids = combined_tasks

                    # Merge extra metadata
                    combined_meta = dict(primary.metadata)
                    for k, v in candidate.metadata.items():
                        if k not in combined_meta:
                            combined_meta[k] = v
                    combined_meta["consolidated_at"] = now.isoformat()
                    combined_meta["consolidated_from_id"] = candidate.id
                    primary.metadata = combined_meta

                    # Retain more informative content if candidate is significantly longer
                    if len(candidate.content) > len(primary.content):
                        primary.content = candidate.content
                        primary.embedding = candidate.embedding

                    primary.updated_at = now

                    # Save updated primary record
                    await self.store.update_memory(primary)

                    # Safely delete candidate record
                    await self.store.delete_memory(candidate.id, scope_id=scope_id)
                    merged_ids.add(candidate.id)
                    merged_count += 1
                    logger.info(
                        "Consolidated memory %s into primary %s (similarity: %.4f)",
                        candidate.id,
                        primary.id,
                        sim,
                    )

        remaining = await self.store.count(scope_id=scope_id)
        return {
            "scope_id": scope_id,
            "expired_deleted": expired_deleted,
            "merged_count": merged_count,
            "remaining_count": remaining,
        }
