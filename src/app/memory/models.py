"""Domain models and data structures for the Memory subsystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Classification of long-term semantic memory records."""

    USER_PREFERENCE = "user_preference"
    TASK_LESSON = "task_lesson"
    SUCCESSFUL_STRATEGY = "successful_strategy"
    DOMAIN_FACT = "domain_fact"
    TASK_SUMMARY = "task_summary"


class MemoryRecord(BaseModel):
    """Represents a persistent long-term semantic memory item."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    scope_id: str = Field(default="default", description="Scope or user identifier")
    content: str = Field(..., description="The factual or procedural knowledge content")
    memory_type: MemoryType = Field(default=MemoryType.DOMAIN_FACT)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    source_task_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def is_expired(self) -> bool:
        """Check if the memory item has exceeded its expiration timestamp."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class MemorySearchResult(BaseModel):
    """Scored memory search result."""

    memory: MemoryRecord
    score: float = Field(..., ge=0.0, le=1.0)
    similarity: float = Field(default=0.0)


class MemoryStats(BaseModel):
    """Aggregated memory statistics."""

    total_memories: int
    memories_by_type: Dict[str, int]
    scope_counts: Dict[str, int]
    storage_backend: str
