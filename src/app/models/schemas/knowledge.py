"""Pydantic schemas for Knowledge and RAG API request/response contracts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentIngestTextRequest(BaseModel):
    """Payload for direct text ingestion."""

    content: str = Field(..., min_length=1, description="Raw text or markdown content to ingest")
    filename: str = Field(default="document.txt", description="Document filename/label")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes")


class DocumentIngestResponse(BaseModel):
    """Response returned upon successful document ingestion."""

    document_id: str
    filename: str
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentListItem(BaseModel):
    """Summary representation of an ingested document."""

    document_id: str
    filename: str
    source: str
    file_type: str
    page_count: Optional[int] = 1
    chunk_count: int
    created_at: Optional[Any] = None


class DocumentListResponse(BaseModel):
    """Response containing list of all indexed documents."""

    documents: List[DocumentListItem]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response returned upon deleting a document."""

    document_id: str
    chunks_deleted: int
    message: str


class KnowledgeSearchRequest(BaseModel):
    """Payload for hybrid or semantic search."""

    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Max number of chunks to retrieve")
    strategy: str = Field(
        default="rrf",
        description="Retrieval strategy: 'rrf' (default), 'weighted', 'semantic', or 'bm25'",
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata key-value filter"
    )


class KnowledgeSearchResultItem(BaseModel):
    """Single retrieval result item."""

    chunk_id: str
    document_id: str
    source: str
    page_number: Optional[int] = None
    score: float
    source_type: str
    content: str


class KnowledgeSearchResponse(BaseModel):
    """Response payload for knowledge search."""

    query: str
    count: int
    results: List[KnowledgeSearchResultItem]


class CitationItem(BaseModel):
    """Structured citation item returned by grounded query."""

    document_id: str
    source: str
    page_number: Optional[int] = None
    chunk_id: str
    score: float
    snippet: str


class KnowledgeQueryRequest(BaseModel):
    """Payload for grounded RAG question answering."""

    query: str = Field(..., min_length=1, description="Question or task requiring grounded context")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    system_prompt: Optional[str] = Field(
        default=None, description="Optional custom system prompt instructing the synthesizer"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filter for context retrieval"
    )


class KnowledgeQueryResponse(BaseModel):
    """Response payload containing synthesized answer and verified citations."""

    query: str
    answer: str
    citations: List[CitationItem]
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class KnowledgeStatsResponse(BaseModel):
    """Knowledge base operational metrics."""

    total_chunks: int
    total_documents: int
    embedding_provider: str
    embedding_dimension: int
    vector_store: str
