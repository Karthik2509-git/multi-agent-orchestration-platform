"""Domain models and data structures for the RAG and Knowledge subsystem."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents an ingested document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """Represents a discrete chunk of a document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class RetrievalResult(BaseModel):
    """Represents a scored chunk returned from hybrid or semantic retrieval."""

    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float
    source_type: str = "hybrid_rrf"


class Citation(BaseModel):
    """Represents a structured source citation for verified RAG outputs."""

    document_id: str
    source: str
    page_number: Optional[int] = None
    chunk_id: str
    score: float
    snippet: str
