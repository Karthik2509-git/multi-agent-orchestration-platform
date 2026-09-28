"""Pydantic schemas for Memory API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.app.memory.models import MemoryRecord, MemoryType


class MemoryCreateRequest(BaseModel):
    """Request schema for creating a new memory record."""

    content: str = Field(..., min_length=1, description="Content to store in memory")
    scope_id: str = Field(default="default", description="Scope or user identifier")
    memory_type: MemoryType = Field(
        default=MemoryType.DOMAIN_FACT, description="Classification of memory"
    )
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance weight between 0.0 and 1.0"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Optional expiration timestamp"
    )
    source_task_id: Optional[str] = Field(default=None, description="Associated task ID")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata dictionary"
    )


class MemoryResponse(BaseModel):
    """Response schema for a single memory record."""

    memory: MemoryRecord


class MemoryListResponse(BaseModel):
    """Response schema for listing memory records."""

    count: int
    memories: List[MemoryRecord]


class MemorySearchRequest(BaseModel):
    """Request schema for semantic memory search."""

    query: str = Field(..., min_length=1, description="Search query")
    scope_id: str = Field(default="default", description="Scope to search within")
    top_k: int = Field(default=3, ge=1, le=50, description="Max memories to retrieve")
    memory_type: Optional[MemoryType] = Field(default=None, description="Optional type filter")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class MemorySearchItem(BaseModel):
    """Individual scored search result."""

    memory: MemoryRecord
    score: float
    similarity: float


class MemorySearchResponse(BaseModel):
    """Response schema for memory search results."""

    query: str
    scope_id: str
    count: int
    results: List[MemorySearchItem]


class MemoryConsolidationResponse(BaseModel):
    """Response schema for memory consolidation operation."""

    scope_id: str
    expired_deleted: int
    merged_count: int
    remaining_count: int


class MemoryStatsResponse(BaseModel):
    """Response schema for memory stats."""

    total_memories: int
    memories_by_type: Dict[str, int]
    scope_counts: Dict[str, int]
    storage_backend: str
