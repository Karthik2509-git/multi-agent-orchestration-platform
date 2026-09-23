"""Knowledge Search Tool connecting agents to the RAG knowledge base via ToolRegistry."""

from typing import Any, Dict, Optional

from src.app.core.logging import get_logger
from src.app.rag.service import RAGService, get_rag_service
from src.app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class KnowledgeSearchTool(BaseTool):
    """Tool allowing agents to search the indexed knowledge base with hybrid retrieval."""

    name: str = "knowledge_search"
    description: str = (
        "Search the internal knowledge base for relevant facts, documents, documentation, "
        "and guidelines using hybrid semantic and keyword retrieval."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or keywords to retrieve information about.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of relevant chunks to retrieve (default: 5).",
                "default": 5,
            },
            "strategy": {
                "type": "string",
                "description": "Strategy: 'rrf' (default), 'weighted', 'semantic', or 'bm25'.",
                "enum": ["rrf", "weighted", "semantic", "bm25"],
                "default": "rrf",
            },
        },
        "required": ["query"],
    }

    def __init__(self, rag_service: Optional[RAGService] = None):
        self._rag_service = rag_service

    @property
    def rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = get_rag_service()
        return self._rag_service

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the knowledge search retrieval query."""
        query = kwargs.get("query")
        if not query or not str(query).strip():
            return ToolResult(
                success=False,
                error="The 'query' parameter is required and cannot be empty.",
            )

        top_k = kwargs.get("top_k", 5)
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 5

        strategy = kwargs.get("strategy", "rrf")

        try:
            results = await self.rag_service.retrieve(
                query=str(query),
                top_k=top_k,
                strategy=strategy,
            )

            formatted_results = []
            for r in results:
                formatted_results.append(
                    {
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "source": r.metadata.get("source", r.metadata.get("filename", "unknown")),
                        "page_number": r.metadata.get("page_number"),
                        "score": r.score,
                        "source_type": r.source_type,
                        "content": r.content,
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "count": len(formatted_results),
                    "results": formatted_results,
                },
            )
        except Exception as e:
            logger.error(f"KnowledgeSearchTool execution failed: {e}")
            return ToolResult(
                success=False,
                error=f"Knowledge search failed: {str(e)}",
            )
