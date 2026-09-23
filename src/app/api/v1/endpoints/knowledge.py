"""RAG & Knowledge base management and query endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.app.core.logging import get_logger
from src.app.models.schemas.knowledge import (
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentIngestTextRequest,
    DocumentListItem,
    DocumentListResponse,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResultItem,
    KnowledgeStatsResponse,
)
from src.app.rag.service import RAGService, get_rag_service

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Text Content",
    description="Ingest raw text or markdown content into the knowledge vector store.",
)
async def ingest_text(
    request: DocumentIngestTextRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentIngestResponse:
    """Ingest raw text into the knowledge base."""
    try:
        doc = await rag_service.ingest_text(
            content=request.content,
            filename=request.filename,
            metadata=request.metadata,
        )
        return DocumentIngestResponse(
            document_id=doc.id,
            filename=doc.filename,
            message="Document text successfully ingested and indexed.",
            created_at=doc.created_at,
        )
    except Exception as e:
        logger.error(f"Failed to ingest document text: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document ingestion failed: {str(e)}",
        ) from e


@router.post(
    "/documents/upload",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest Document File",
    description="Upload a .txt, .md, or .pdf file for chunking, embedding, and vector indexing.",
)
async def upload_document_file(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentIngestResponse:
    """Upload and ingest a file into the knowledge base."""
    try:
        content_bytes = await file.read()
        filename = file.filename or "uploaded_file"
        doc = await rag_service.ingest_file(
            file_path_or_buffer=content_bytes,
            filename=filename,
        )
        return DocumentIngestResponse(
            document_id=doc.id,
            filename=doc.filename,
            message=f"File '{filename}' successfully ingested and indexed.",
            created_at=doc.created_at,
        )
    except Exception as e:
        logger.error(f"Failed to upload and ingest file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File ingestion failed: {str(e)}",
        ) from e


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Ingested Documents",
    description="List all distinct documents indexed in the knowledge base.",
)
async def list_documents(
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentListResponse:
    """List indexed documents."""
    docs = await rag_service.list_documents()
    items = [
        DocumentListItem(
            document_id=d["document_id"],
            filename=d.get("filename", "unknown"),
            source=d.get("source", "unknown"),
            file_type=d.get("file_type", ".txt"),
            page_count=d.get("page_count", 1),
            chunk_count=d.get("chunk_count", 0),
            created_at=d.get("created_at"),
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Delete all chunks and vectors associated with a document ID.",
)
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentDeleteResponse:
    """Delete a document from the knowledge base."""
    count = await rag_service.delete_document(document_id)
    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DocumentDeleteResponse(
        document_id=document_id,
        chunks_deleted=count,
        message=f"Successfully deleted {count} chunks for document '{document_id}'.",
    )


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Knowledge Base",
    description="Retrieve relevant document chunks using hybrid RRF, BM25, or dense search.",
)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> KnowledgeSearchResponse:
    """Execute hybrid retrieval search."""
    try:
        results = await rag_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            strategy=request.strategy,
        )
        items = [
            KnowledgeSearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                source=r.metadata.get("source", r.metadata.get("filename", "unknown")),
                page_number=r.metadata.get("page_number"),
                score=r.score,
                source_type=r.source_type,
                content=r.content,
            )
            for r in results
        ]
        return KnowledgeSearchResponse(
            query=request.query,
            count=len(items),
            results=items,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e


@router.post(
    "/query",
    response_model=KnowledgeQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded Question Answering",
    description="Answer questions grounded in retrieved documents with verified source citations.",
)
async def query_knowledge(
    request: KnowledgeQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> KnowledgeQueryResponse:
    """Execute grounded RAG query with citations."""
    try:
        response_dict = await rag_service.query(
            query_text=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            system_prompt=request.system_prompt,
        )
        return KnowledgeQueryResponse(**response_dict)
    except Exception as e:
        logger.error(f"Knowledge query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge query failed: {str(e)}",
        ) from e


@router.get(
    "/stats",
    response_model=KnowledgeStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Knowledge Base Statistics",
    description="Return summary metrics for the vector store and embedding provider.",
)
async def get_knowledge_stats(
    rag_service: RAGService = Depends(get_rag_service),
) -> KnowledgeStatsResponse:
    """Return knowledge base stats."""
    stats = await rag_service.get_stats()
    return KnowledgeStatsResponse(**stats)
