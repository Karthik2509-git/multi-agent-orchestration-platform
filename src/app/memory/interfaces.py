"""Abstract interfaces for long-term memory stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.app.memory.models import MemoryRecord, MemorySearchResult, MemoryType


class MemoryStore(ABC):
    """Abstract interface for long-term semantic memory storage."""

    @abstractmethod
    async def add_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Add or upsert a memory record."""
        pass

    @abstractmethod
    async def get_memory(
        self, memory_id: str, scope_id: Optional[str] = None
    ) -> Optional[MemoryRecord]:
        """Retrieve a specific memory record by ID with optional scope validation."""
        pass

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        scope_id: str = "default",
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        """Search memory records using semantic similarity with strict scope isolation."""
        pass

    @abstractmethod
    async def list_memories(
        self,
        scope_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List memory records with filtering and pagination."""
        pass

    @abstractmethod
    async def update_memory(self, record: MemoryRecord) -> Optional[MemoryRecord]:
        """Update an existing memory record."""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str, scope_id: Optional[str] = None) -> bool:
        """Delete a memory record by ID with optional scope validation."""
        pass

    @abstractmethod
    async def count(self, scope_id: Optional[str] = None) -> int:
        """Return the number of stored memories."""
        pass
