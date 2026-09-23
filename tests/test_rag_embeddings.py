"""Tests for embedding providers."""

import pytest

from src.app.core.config import Settings
from src.app.rag.embeddings import (
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)


@pytest.mark.asyncio
async def test_mock_embedding_provider():
    """Test MockEmbeddingProvider returns deterministic normalized vectors."""
    provider = MockEmbeddingProvider(dimension=32)
    assert provider.dimension == 32

    vec1 = await provider.embed_query("machine learning")
    vec2 = await provider.embed_query("machine learning")
    vec3 = await provider.embed_query("deep neural networks")

    assert len(vec1) == 32
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Distinct texts produce distinct vectors

    docs = await provider.embed_documents(["doc 1", "doc 2"])
    assert len(docs) == 2
    assert len(docs[0]) == 32
    assert len(docs[1]) == 32


@pytest.mark.asyncio
async def test_local_embedding_provider():
    """Test LocalEmbeddingProvider dimension and embedding execution."""
    provider = LocalEmbeddingProvider()
    assert provider.dimension == 384

    vec = await provider.embed_query("antigravity orchestration")
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

    batch = await provider.embed_documents(["document one", "document two"])
    assert len(batch) == 2
    assert len(batch[0]) == 384


def test_embedding_factory():
    """Test factory resolution for mock, local, and default configurations."""
    mock_prov = get_embedding_provider("mock")
    assert isinstance(mock_prov, MockEmbeddingProvider)

    local_prov = get_embedding_provider("local")
    assert isinstance(local_prov, LocalEmbeddingProvider)

    custom_settings = Settings(rag_embedding_provider="mock")
    prov_from_settings = get_embedding_provider(settings=custom_settings)
    assert isinstance(prov_from_settings, MockEmbeddingProvider)
