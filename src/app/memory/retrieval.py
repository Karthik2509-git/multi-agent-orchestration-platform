"""Memory retrieval and ranking engine for supervisor planning."""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.app.core.logging import get_logger
from src.app.memory.interfaces import MemoryStore
from src.app.memory.models import MemoryRecord, MemorySearchResult, MemoryType

logger = get_logger(__name__)


class MemoryRetriever:
    """Retrieves and ranks relevant long-term memories for agent planning."""

    def __init__(
        self,
        store: MemoryStore,
        similarity_weight: float = 0.65,
        importance_weight: float = 0.25,
        recency_weight: float = 0.10,
        similarity_threshold: float = 0.0,
    ):
        self.store = store
        self.similarity_weight = similarity_weight
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.similarity_threshold = similarity_threshold

    def calculate_recency_score(
        self, record: MemoryRecord, now: Optional[datetime] = None
    ) -> float:
        """Calculate recency factor in range (0.0, 1.0] based on days elapsed since update."""
        current_time = now or datetime.now(timezone.utc)
        elapsed_seconds = max(0.0, (current_time - record.updated_at).total_seconds())
        days_elapsed = elapsed_seconds / 86400.0
        # Exponential decay factor with half-life ~ 14 days
        return math.exp(-0.05 * days_elapsed)

    def compute_composite_score(
        self,
        similarity: float,
        record: MemoryRecord,
        now: Optional[datetime] = None,
    ) -> float:
        """Compute ranking score: 0.65*similarity + 0.25*importance + 0.10*recency."""
        recency = self.calculate_recency_score(record, now)
        importance = max(0.0, min(1.0, record.importance))
        sim = max(0.0, min(1.0, similarity))

        score = (
            self.similarity_weight * sim
            + self.importance_weight * importance
            + self.recency_weight * recency
        )
        return round(score, 5)

    async def retrieve(
        self,
        query: str,
        scope_id: str = "default",
        top_k: int = 3,
        memory_type: Optional[MemoryType] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        track_access: bool = True,
    ) -> List[MemorySearchResult]:
        """Retrieve and rank relevant memories for a query within a scope."""
        # Query candidate pool with top_k * 3 to allow composite reranking
        candidates = await self.store.search_memories(
            query=query,
            scope_id=scope_id,
            top_k=max(10, top_k * 3),
            memory_type=memory_type,
            filter_metadata=filter_metadata,
        )

        if not candidates:
            return []

        now = datetime.now(timezone.utc)
        scored_results: List[MemorySearchResult] = []

        for candidate in candidates:
            if candidate.similarity < self.similarity_threshold:
                continue

            composite = self.compute_composite_score(
                similarity=candidate.similarity,
                record=candidate.memory,
                now=now,
            )

            scored_results.append(
                MemorySearchResult(
                    memory=candidate.memory,
                    score=composite,
                    similarity=candidate.similarity,
                )
            )

        # Sort descending by composite score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        final_results = scored_results[:top_k]

        # Update access tracking metadata if enabled
        if track_access:
            for res in final_results:
                res.memory.access_count += 1
                res.memory.last_accessed_at = now
                try:
                    await self.store.update_memory(res.memory)
                except Exception as e:
                    logger.warning(
                        "Failed to update access metadata for memory %s: %s", res.memory.id, e
                    )

        return final_results

    def format_for_planning(self, search_results: List[MemorySearchResult]) -> str:
        """Format retrieved memories into a clean bulleted list for supervisor planning prompt."""
        if not search_results:
            return ""

        formatted_lines = ["Relevant Memories & Past Context:"]
        for res in search_results:
            mem = res.memory
            type_label = mem.memory_type.value.replace("_", " ").title()
            formatted_lines.append(
                f"- [{type_label}] {mem.content} (importance: {mem.importance:.2f})"
            )

        return "\n".join(formatted_lines)
