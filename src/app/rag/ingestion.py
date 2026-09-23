"""Document loaders and ingestion pipelines for text, markdown, and PDF files."""

import io
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union
from uuid import uuid4

from src.app.core.config import get_settings
from src.app.core.logging import get_logger
from src.app.rag.chunking import RecursiveChunker
from src.app.rag.models import Document, DocumentChunk

logger = get_logger(__name__)


class DocumentLoader:
    """Loads and extracts text and metadata from files (.txt, .md, .pdf)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".markdown", ".json", ".csv"}

    def __init__(
        self,
        chunker: Optional[RecursiveChunker] = None,
        max_file_size_bytes: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunker = chunker or RecursiveChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        self.max_file_size_bytes = max_file_size_bytes or settings.rag_max_file_size_bytes

    def load_from_text(
        self,
        content: str,
        filename: str = "document.txt",
        metadata: Optional[dict] = None,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Ingest plain text or markdown directly."""
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty.")

        doc_id = str(uuid4())
        doc_metadata = {
            "source": filename,
            "filename": filename,
            "file_type": Path(filename).suffix.lower() or ".txt",
            **(metadata or {}),
        }

        document = Document(
            id=doc_id,
            filename=filename,
            content=content,
            metadata=doc_metadata,
        )

        base_meta = {
            "source": filename,
            "filename": filename,
            "page_number": 1,
            **(metadata or {}),
        }
        chunks = self.chunker.chunk_document(
            document_id=doc_id,
            content=content,
            base_metadata=base_meta,
        )

        return document, chunks

    def load_from_file(
        self,
        file_path_or_buffer: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Load and extract text from a file path or binary buffer (txt, md, pdf)."""
        if isinstance(file_path_or_buffer, (str, Path)):
            path = Path(file_path_or_buffer)
            name = filename or path.name
            size = path.stat().st_size
            if size > self.max_file_size_bytes:
                raise ValueError(
                    f"File size ({size} bytes) exceeds limit ({self.max_file_size_bytes} bytes)."
                )
            with open(path, "rb") as f:
                content_bytes = f.read()
        elif isinstance(file_path_or_buffer, bytes):
            name = filename or "uploaded_file"
            content_bytes = file_path_or_buffer
            if len(content_bytes) > self.max_file_size_bytes:
                raise ValueError(
                    f"File size ({len(content_bytes)} bytes) exceeds limit "
                    f"({self.max_file_size_bytes} bytes)."
                )
        else:
            name = filename or "uploaded_file"
            content_bytes = file_path_or_buffer.read()
            if len(content_bytes) > self.max_file_size_bytes:
                raise ValueError(
                    f"File size ({len(content_bytes)} bytes) exceeds limit "
                    f"({self.max_file_size_bytes} bytes)."
                )

        if not content_bytes:
            raise ValueError("File content cannot be empty.")

        ext = Path(name).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            fmts = sorted(self.SUPPORTED_EXTENSIONS)
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {fmts}")

        if ext == ".pdf":
            return self._load_pdf(content_bytes, name, metadata)
        else:
            return self._load_text_file(content_bytes, name, ext, metadata)

    def _load_text_file(
        self,
        content_bytes: bytes,
        filename: str,
        ext: str,
        metadata: Optional[dict] = None,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Decode and chunk plain text/markdown file."""
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1", errors="replace")

        if not text.strip():
            raise ValueError("File contains no readable text.")

        return self.load_from_text(
            content=text,
            filename=filename,
            metadata=metadata,
        )

    def _load_pdf(
        self,
        content_bytes: bytes,
        filename: str,
        metadata: Optional[dict] = None,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Extract text page-by-page from PDF and preserve page metadata."""
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "pypdf is required for PDF ingestion. Install with `pip install pypdf`."
            ) from e

        reader = PdfReader(io.BytesIO(content_bytes))
        total_pages = len(reader.pages)

        if total_pages == 0:
            raise ValueError("PDF contains no pages.")

        page_texts: List[Tuple[int, str]] = []
        full_text_parts: List[str] = []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append((page_num, extracted.strip()))
                full_text_parts.append(f"--- Page {page_num} ---\n{extracted.strip()}")

        if not full_text_parts:
            raise ValueError("PDF contains no extractable text.")

        full_content = "\n\n".join(full_text_parts)
        doc_id = str(uuid4())

        doc_meta = {
            "source": filename,
            "filename": filename,
            "file_type": ".pdf",
            "page_count": total_pages,
            **(metadata or {}),
        }

        document = Document(
            id=doc_id,
            filename=filename,
            content=full_content,
            metadata=doc_meta,
        )

        all_chunks: List[DocumentChunk] = []
        chunk_offset = 0

        for page_num, page_content in page_texts:
            page_meta = {
                "source": filename,
                "filename": filename,
                "file_type": ".pdf",
                "page_number": page_num,
                "page_count": total_pages,
                **(metadata or {}),
            }
            page_chunks = self.chunker.chunk_document(
                document_id=doc_id,
                content=page_content,
                base_metadata=page_meta,
            )
            for c in page_chunks:
                c.index = chunk_offset
                c.metadata["chunk_index"] = chunk_offset
                chunk_offset += 1
                all_chunks.append(c)

        return document, all_chunks
