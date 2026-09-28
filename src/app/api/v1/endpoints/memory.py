"""REST API endpoints for Long-Term Semantic Memory management."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.app.core.logging import get_logger
from src.app.memory.models import MemoryType
from src.app.memory.service import MemoryService, get_memory_service
from src.app.models.schemas.memory import (
    MemoryConsolidationResponse,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    MemorySearchItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStatsResponse,
)

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Memory Record",
    description="Add a new long-term semantic memory item within a given scope.",
)
async def create_memory(
    request: MemoryCreateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    try:
        record = await memory_service.add_memory(
            content=request.content,
            scope_id=request.scope_id,
            memory_type=request.memory_type,
            importance=request.importance,
            expires_at=request.expires_at,
            source_task_id=request.source_task_id,
            metadata=request.metadata,
        )
        return MemoryResponse(memory=record)
    except Exception as e:
        logger.error("Failed to create memory record: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory record: {str(e)}",
        )


@router.get(
    "",
    response_model=MemoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Memories",
    description="List stored memory records with optional scope and type filtering.",
)
async def list_memories(
    scope_id: Optional[str] = Query(default=None, description="Scope identifier filter"),
    memory_type: Optional[MemoryType] = Query(
        default=None, description="Memory classification filter"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryListResponse:
    records = await memory_service.list_memories(
        scope_id=scope_id,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
    )
    return MemoryListResponse(count=len(records), memories=records)


@router.get(
    "/stats",
    response_model=MemoryStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Memory Stats",
    description="Retrieve aggregated statistics about stored memories.",
)
async def get_memory_stats(
    scope_id: Optional[str] = Query(default=None, description="Scope filter"),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryStatsResponse:
    stats = await memory_service.get_stats(scope_id=scope_id)
    return MemoryStatsResponse(
        total_memories=stats.total_memories,
        memories_by_type=stats.memories_by_type,
        scope_counts=stats.scope_counts,
        storage_backend=stats.storage_backend,
    )


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Memory by ID",
    description="Retrieve a single memory record by unique identifier.",
)
async def get_memory_by_id(
    memory_id: str,
    scope_id: Optional[str] = Query(default=None, description="Scope validation"),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    record = await memory_service.get_memory(memory_id, scope_id=scope_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory record with ID '{memory_id}' not found.",
        )
    return MemoryResponse(memory=record)


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Memory",
    description="Delete a memory record by ID with optional scope validation.",
)
async def delete_memory(
    memory_id: str,
    scope_id: Optional[str] = Query(default=None, description="Scope validation"),
    memory_service: MemoryService = Depends(get_memory_service),
) -> dict:
    success = await memory_service.delete_memory(memory_id, scope_id=scope_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory record with ID '{memory_id}' not found.",
        )
    return {"status": "deleted", "memory_id": memory_id}


@router.post(
    "/search",
    response_model=MemorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Memories",
    description="Search and rank memories using semantic similarity and composite scoring.",
)
async def search_memories(
    request: MemorySearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemorySearchResponse:
    results = await memory_service.search_memories(
        query=request.query,
        scope_id=request.scope_id,
        top_k=request.top_k,
        memory_type=request.memory_type,
        filter_metadata=request.filter_metadata,
    )
    items = [
        MemorySearchItem(memory=r.memory, score=r.score, similarity=r.similarity) for r in results
    ]
    return MemorySearchResponse(
        query=request.query,
        scope_id=request.scope_id,
        count=len(items),
        results=items,
    )


@router.post(
    "/consolidate",
    response_model=MemoryConsolidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Consolidate Memories",
    description="Merge duplicate memories and remove expired items within a scope.",
)
async def consolidate_memories(
    scope_id: str = Query(default="default", description="Scope identifier"),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryConsolidationResponse:
    result = await memory_service.consolidate(scope_id=scope_id)
    return MemoryConsolidationResponse(
        scope_id=result["scope_id"],
        expired_deleted=result["expired_deleted"],
        merged_count=result["merged_count"],
        remaining_count=result["remaining_count"],
    )
