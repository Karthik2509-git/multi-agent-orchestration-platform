"""Deterministic offline evaluation fixtures and benchmark dataset."""

from typing import List

from src.app.evaluation.models import EvaluationCase, EvaluationDataset
from src.app.rag.models import RetrievalResult


def _create_mock_result(chunk_id: str, content: str = "", score: float = 0.5) -> RetrievalResult:
    """Helper to create dummy RetrievalResult for fixture cases."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content or f"Sample content for chunk {chunk_id}",
        score=score,
        source_type="hybrid_rrf",
    )


def create_deterministic_benchmark_cases() -> List[EvaluationCase]:
    """Create representative deterministic evaluation cases covering all core scenarios.

    Scenarios included:
    1. Rank 1 relevant result (single relevant chunk retrieved at position 1)
    2. Later rank relevant result (single relevant chunk retrieved at position 3)
    3. No relevant result retrieved (relevant chunk missing from top K)
    4. Multiple relevant chunks (3 relevant, 2 retrieved in top 5, 1 missing)
    5. Fewer than K results retrieved (K=5, but only 2 chunks retrieved, 1 relevant)
    6. Empty relevant chunk set (0 ground-truth relevant chunks defined)
    7. Empty retrieved results (no chunks returned by retriever)
    8. Duplicate retrieved chunk IDs (retriever returns duplicated ID in top results)
    9. Fully grounded assertion (all assertion tokens present in retrieved context)
    10. Ungrounded assertion (hallucinated content unsupported by context)
    """
    cases = [
        # 1. Relevant result at rank 1
        EvaluationCase(
            case_id="case-rank-1",
            query="What is hybrid search in RAG?",
            relevant_chunk_ids=["chunk-hybrid-01"],
            retrieved_results=[
                _create_mock_result(
                    "chunk-hybrid-01",
                    "Hybrid search combines dense vector search with sparse BM25 keyword matching.",
                    0.95,
                ),
                _create_mock_result(
                    "chunk-dense-02", "Dense vector embeddings capture semantic meaning.", 0.80
                ),
                _create_mock_result(
                    "chunk-bm25-03", "BM25 keyword search scores exact token matches.", 0.70
                ),
            ],
            reference_answer="Hybrid search combines dense and sparse search.",
            generated_answer=(
                "Hybrid search combines dense vector search with sparse BM25 keyword matching."
            ),
            context=(
                "Hybrid search combines dense vector search with sparse BM25 keyword matching."
            ),
            metadata={"scenario": "rank_1_relevant"},
        ),
        # 2. Relevant result at a later rank (rank 3)
        EvaluationCase(
            case_id="case-rank-3",
            query="How does reciprocal rank fusion work?",
            relevant_chunk_ids=["chunk-rrf-core"],
            retrieved_results=[
                _create_mock_result("chunk-unrelated-01", "Unrelated retrieval result.", 0.88),
                _create_mock_result("chunk-unrelated-02", "Another tangential document.", 0.82),
                _create_mock_result(
                    "chunk-rrf-core",
                    "Reciprocal Rank Fusion merges rankings using 1 / (60 + rank).",
                    0.79,
                ),
                _create_mock_result("chunk-other-04", "Vector indices are queried first.", 0.65),
            ],
            reference_answer="RRF merges rankings by reciprocal rank scores.",
            generated_answer="RRF combines rankings using reciprocal rank formula.",
            context="Reciprocal Rank Fusion merges rankings using 1 / (60 + rank).",
            metadata={"scenario": "later_rank_relevant"},
        ),
        # 3. No relevant result retrieved
        EvaluationCase(
            case_id="case-no-relevant",
            query="What are quantum computing error correction codes?",
            relevant_chunk_ids=["chunk-quantum-surface-codes"],
            retrieved_results=[
                _create_mock_result(
                    "chunk-classical-01", "Classical parity checks detect single bit errors.", 0.60
                ),
                _create_mock_result(
                    "chunk-classical-02", "Hamming codes provide linear error correction.", 0.55
                ),
            ],
            reference_answer=(
                "Quantum error correction uses surface codes and topological protection."
            ),
            generated_answer="Quantum error correction uses surface codes.",
            context=(
                "Classical parity checks detect single bit errors. "
                "Hamming codes provide linear error correction."
            ),
            metadata={"scenario": "no_relevant_retrieved"},
        ),
        # 4. Multiple relevant chunks
        EvaluationCase(
            case_id="case-multiple-relevant",
            query="Explain the components of the LangGraph multi-agent architecture.",
            relevant_chunk_ids=[
                "chunk-agent-supervisor",
                "chunk-agent-specialist",
                "chunk-agent-state",
            ],
            retrieved_results=[
                _create_mock_result(
                    "chunk-agent-supervisor",
                    "The supervisor router routes tasks to specialist agents.",
                    0.92,
                ),
                _create_mock_result(
                    "chunk-unrelated-05", "PostgreSQL provides relational storage.", 0.85
                ),
                _create_mock_result(
                    "chunk-agent-specialist",
                    "Specialist agents execute specific domain tasks.",
                    0.78,
                ),
                _create_mock_result(
                    "chunk-unrelated-06", "HTTP API endpoints handle ingress requests.", 0.71
                ),
                _create_mock_result(
                    "chunk-unrelated-07", "Docker compose orchestrates containers.", 0.65
                ),
            ],
            reference_answer="LangGraph uses a supervisor, specialists, and shared state.",
            generated_answer="The supervisor routes tasks to specialist agents.",
            context=(
                "The supervisor router routes tasks to specialist agents. "
                "Specialist agents execute specific domain tasks."
            ),
            metadata={"scenario": "multiple_relevant_chunks"},
        ),
        # 5. Fewer than K results retrieved (e.g. K=5, only 2 retrieved, 1 relevant)
        EvaluationCase(
            case_id="case-fewer-than-k",
            query="What is chunk overlap in text splitting?",
            relevant_chunk_ids=["chunk-overlap-01"],
            retrieved_results=[
                _create_mock_result(
                    "chunk-overlap-01",
                    "Chunk overlap prevents losing context at chunk boundaries.",
                    0.90,
                ),
                _create_mock_result(
                    "chunk-token-02", "Token length determines splitting positions.", 0.75
                ),
            ],
            reference_answer="Chunk overlap retains context across adjacent text chunks.",
            generated_answer="Chunk overlap retains boundary context.",
            context="Chunk overlap prevents losing context at chunk boundaries.",
            metadata={"scenario": "fewer_than_k"},
        ),
        # 6. Empty ground-truth relevant set
        EvaluationCase(
            case_id="case-empty-relevant",
            query="Describe an unindexed hypothetical concept.",
            relevant_chunk_ids=[],
            retrieved_results=[
                _create_mock_result("chunk-random-01", "Some random text chunk.", 0.50),
            ],
            metadata={"scenario": "empty_relevant_set"},
        ),
        # 7. Empty retrieved results
        EvaluationCase(
            case_id="case-empty-retrieved",
            query="Query that returns nothing from index.",
            relevant_chunk_ids=["chunk-needed-01"],
            retrieved_results=[],
            metadata={"scenario": "empty_retrieved_results"},
        ),
        # 8. Duplicate retrieved chunk IDs
        EvaluationCase(
            case_id="case-duplicate-retrieved",
            query="Query with duplicated chunk returns.",
            relevant_chunk_ids=["chunk-dup-01"],
            retrieved_results=[
                _create_mock_result("chunk-dup-01", "Content about duplication.", 0.90),
                _create_mock_result("chunk-dup-01", "Content about duplication.", 0.85),
                _create_mock_result("chunk-other-02", "Other content.", 0.70),
            ],
            metadata={"scenario": "duplicate_retrieved_results"},
        ),
        # 9. Grounded assertion (full lexical overlap)
        EvaluationCase(
            case_id="case-grounded-overlap",
            query="What is the capital of France?",
            relevant_chunk_ids=["chunk-france-01"],
            retrieved_results=[
                _create_mock_result("chunk-france-01", "Paris is the capital of France.", 0.99),
            ],
            generated_answer="Paris is the capital of France.",
            context="Paris is the capital of France.",
            metadata={"scenario": "grounded_assertion"},
        ),
        # 10. Ungrounded assertion (zero lexical overlap)
        EvaluationCase(
            case_id="case-ungrounded-overlap",
            query="What is the population of Mars?",
            relevant_chunk_ids=["chunk-mars-01"],
            retrieved_results=[
                _create_mock_result(
                    "chunk-mars-01", "Mars is an uninhabited planet with a thin atmosphere.", 0.90
                ),
            ],
            generated_answer=(
                "Ten million human beings live permanently underground in robotic biodomes."
            ),
            context="Mars is an uninhabited planet with a thin atmosphere.",
            metadata={"scenario": "ungrounded_assertion"},
        ),
    ]
    return cases


def get_deterministic_benchmark_dataset() -> EvaluationDataset:
    """Return standard deterministic benchmark dataset for offline evaluation."""
    cases = create_deterministic_benchmark_cases()
    return EvaluationDataset(
        name="rag_deterministic_benchmark_v1",
        description=(
            "Standard offline benchmark dataset covering RAG retrieval "
            "and lexical groundedness edge cases."
        ),
        cases=cases,
        metadata={"version": "1.0.0", "case_count": len(cases)},
    )
