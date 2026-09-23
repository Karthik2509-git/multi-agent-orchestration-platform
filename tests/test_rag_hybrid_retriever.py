"""Tests for BM25Scorer and HybridRetriever with RRF fusion."""

import pytest

from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.hybrid_retriever import BM25Scorer, HybridRetriever
from src.app.rag.models import DocumentChunk


def test_bm25_scorer_tokenization():
    """Test BM25 tokenization strips stop words and punctuation."""
    scorer = BM25Scorer()
    tokens = scorer.tokenize("The quick brown fox jumps over the lazy dog!")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_bm25_scoring_ranking():
    """Test BM25 ranks documents containing matching keywords higher."""
    scorer = BM25Scorer()
    chunks = [
        DocumentChunk(
            id="c1",
            document_id="d1",
            content="PostgreSQL is a powerful open-source relational database system.",
            index=0,
        ),
        DocumentChunk(
            id="c2",
            document_id="d2",
            content="Redis is an in-memory data store used as a cache and broker.",
            index=0,
        ),
        DocumentChunk(
            id="c3",
            document_id="d3",
            content="FastAPI is a modern web framework for Python APIs.",
            index=0,
        ),
    ]

    results = scorer.score("relational database postgresql", chunks, top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_hybrid_retriever_rrf():
    """Test HybridRetriever executes RRF fusion correctly."""
    store = ChromaVectorStore(collection_name="test_rrf_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)

    # Ingest chunks with distinct embeddings
    c1_emb = await embedder.embed_query("quantum computing qubits superposition")
    c2_emb = await embedder.embed_query("classical computing silicon transistors")

    chunks = [
        DocumentChunk(
            id="qc-1",
            document_id="doc-qc",
            content="Quantum computing uses quantum bits or qubits to perform computations.",
            index=0,
            metadata={"domain": "quantum"},
            embedding=c1_emb,
        ),
        DocumentChunk(
            id="cc-1",
            document_id="doc-cc",
            content="Classical computers process information using binary bits of 0 and 1.",
            index=0,
            metadata={"domain": "classical"},
            embedding=c2_emb,
        ),
    ]
    await store.add_chunks(chunks)

    retriever = HybridRetriever(
        vector_store=store,
        embedding_provider=embedder,
        fusion_strategy="rrf",
    )

    # Query matching quantum
    results = await retriever.retrieve(
        query="qubits and quantum computing",
        top_k=2,
    )

    assert len(results) >= 1
    assert results[0].chunk_id == "qc-1"
    assert results[0].source_type == "hybrid_rrf"
    assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_hybrid_retriever_strategies():
    """Test selecting different strategies: weighted, semantic, and bm25."""
    store = ChromaVectorStore(collection_name="test_strategies_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    c_emb = await embedder.embed_query("agent orchestration")

    chunk = DocumentChunk(
        id="agent-chunk",
        document_id="doc-agent",
        content="Multi-agent orchestration allows multiple specialized agents to collaborate.",
        index=0,
        embedding=c_emb,
    )
    await store.add_chunks([chunk])

    retriever = HybridRetriever(
        vector_store=store,
        embedding_provider=embedder,
    )

    # Test semantic strategy
    sem_res = await retriever.retrieve("agent orchestration", strategy="semantic")
    assert len(sem_res) == 1
    assert sem_res[0].source_type == "semantic"

    # Test bm25 strategy
    bm25_res = await retriever.retrieve("specialized agents", strategy="bm25")
    assert len(bm25_res) == 1
    assert bm25_res[0].source_type == "bm25"

    # Test weighted strategy
    wt_res = await retriever.retrieve("orchestration", strategy="weighted")
    assert len(wt_res) == 1
    assert wt_res[0].source_type == "hybrid_weighted"
