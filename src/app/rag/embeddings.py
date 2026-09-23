"""Embedding provider abstractions and implementations for RAG."""

import asyncio
import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Optional

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate an embedding vector for a search query string."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of document texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this provider."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, lightweight embedding provider for testing without external downloads."""

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_text(self, text: str) -> List[float]:
        """Generate a deterministic unit-normalized float vector based on text SHA256 hash."""
        if not text:
            return [0.0] * self._dimension

        # Derive pseudo-random deterministic floats from hash bytes
        hasher = hashlib.sha256(text.encode("utf-8"))
        digest = hasher.digest()

        vec = []
        for i in range(self._dimension):
            byte_idx = i % len(digest)
            # Normalize byte to range [-1.0, 1.0]
            val = ((digest[byte_idx] + i * 17) % 256 - 128) / 128.0
            vec.append(val)

        # Unit-normalize vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local ONNX-based embedding provider using Chroma's default all-MiniLM-L6-v2."""

    def __init__(self):
        self._dimension = 384
        self._ef = None

    def _get_ef(self):
        if self._ef is None:
            try:
                from chromadb.utils import embedding_functions

                self._ef = embedding_functions.DefaultEmbeddingFunction()
                logger.info("Initialized LocalEmbeddingProvider with all-MiniLM-L6-v2 (ONNX).")
            except Exception as e:
                logger.error(f"Failed to initialize DefaultEmbeddingFunction: {e}")
                raise RuntimeError(
                    f"LocalEmbeddingProvider could not load Chroma ONNX model: {e}"
                ) from e
        return self._ef

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> List[float]:
        ef = self._get_ef()
        embeddings = await asyncio.to_thread(ef, [text])
        return [float(x) for x in embeddings[0]]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        ef = self._get_ef()
        embeddings = await asyncio.to_thread(ef, texts)
        return [[float(x) for x in emb] for emb in embeddings]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or get_settings().openai_api_key
        self.model = model
        self._dimension = 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> List[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.embeddings.create(input=texts, model=self.model)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise RuntimeError(f"OpenAI embedding generation error: {e}") from e


def get_embedding_provider(
    provider_type: Optional[str] = None, settings: Optional[Settings] = None
) -> EmbeddingProvider:
    """Factory function returning the configured EmbeddingProvider instance."""
    app_settings = settings or get_settings()
    selected = (provider_type or app_settings.rag_embedding_provider).lower().strip()

    if selected == "mock":
        return MockEmbeddingProvider()
    elif selected == "local":
        return LocalEmbeddingProvider()
    elif selected == "openai":
        return OpenAIEmbeddingProvider(api_key=app_settings.openai_api_key)
    else:
        logger.warning(
            f"Unknown embedding provider '{selected}', falling back to LocalEmbeddingProvider"
        )
        return LocalEmbeddingProvider()
