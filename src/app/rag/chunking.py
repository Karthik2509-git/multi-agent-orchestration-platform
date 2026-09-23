"""Text chunking strategies for RAG document ingestion."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.app.rag.models import DocumentChunk


class RecursiveChunker:
    """Recursively splits text into chunks using a hierarchy of natural separators."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using the given separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator and s:
                good_splits.append(s)
            elif not separator and s:
                good_splits.append(s)

        current_chunk: List[str] = []
        total_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk and separator else 0)
            if total_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                total_len += piece_len
            else:
                if current_chunk:
                    joined = separator.join(current_chunk)
                    if len(joined) > self.chunk_size and new_separators:
                        final_chunks.extend(self._split_text(joined, new_separators))
                    else:
                        final_chunks.append(joined)

                    overlap_parts: List[str] = []
                    overlap_len = 0
                    for part in reversed(current_chunk):
                        part_len = len(part) + (
                            len(separator) if overlap_parts and separator else 0
                        )
                        if overlap_len + part_len <= self.chunk_overlap:
                            overlap_parts.insert(0, part)
                            overlap_len += part_len
                        else:
                            break

                    current_chunk = list(overlap_parts)
                    total_len = overlap_len

                piece_len_with_sep = len(piece) + (
                    len(separator) if current_chunk and separator else 0
                )
                if piece_len_with_sep <= self.chunk_size:
                    current_chunk.append(piece)
                    total_len += piece_len_with_sep
                else:
                    if new_separators:
                        sub_chunks = self._split_text(piece, new_separators)
                        final_chunks.extend(sub_chunks)
                    else:
                        for i in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                            final_chunks.append(piece[i : i + self.chunk_size])
                    current_chunk = []
                    total_len = 0

        if current_chunk:
            joined = separator.join(current_chunk)
            if len(joined) > self.chunk_size and new_separators:
                final_chunks.extend(self._split_text(joined, new_separators))
            else:
                final_chunks.append(joined)

        return [c.strip() for c in final_chunks if c.strip()]

    def chunk_document(
        self,
        document_id: str,
        content: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split document content into structured DocumentChunk objects."""
        if not content or not content.strip():
            return []

        raw_chunks = self._split_text(content.strip(), self.separators)
        meta = base_metadata.copy() if base_metadata else {}

        document_chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_metadata = {
                **meta,
                "document_id": document_id,
                "chunk_index": idx,
                "char_length": len(chunk_text),
            }
            document_chunks.append(
                DocumentChunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    content=chunk_text,
                    index=idx,
                    metadata=chunk_metadata,
                )
            )

        return document_chunks
