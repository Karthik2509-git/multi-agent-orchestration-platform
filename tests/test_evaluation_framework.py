"""Comprehensive test suite for Phase 7 Milestone 4: Evaluation Framework."""

import pytest

from src.app.core.config import Settings
from src.app.evaluation.evaluator import RAGEvaluator
from src.app.evaluation.fixtures import (
    create_deterministic_benchmark_cases,
    get_deterministic_benchmark_dataset,
)
from src.app.evaluation.judge import BaseLLMJudge, LLMJudge, MockLLMJudge
from src.app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    calculate_context_assertion_overlap,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_retrieval_metrics,
)
from src.app.evaluation.models import (
    EvaluationCase,
    GroundTruth,
    JudgeEvaluationResult,
)
from src.app.llm.providers.mock import MockLLMProvider
from src.app.models.schemas.llm import LLMResponse
from src.app.observability import trace_span
from src.app.rag.chroma_store import ChromaVectorStore
from src.app.rag.embeddings import MockEmbeddingProvider
from src.app.rag.service import RAGService

# ============================================================================
# 1. GROUND TRUTH MODEL TESTS
# ============================================================================


def test_ground_truth_valid():
    """Test creating valid GroundTruth with relevant chunk IDs."""
    gt = GroundTruth(
        query="What is hybrid search?",
        relevant_chunk_ids=["chunk-01", "chunk-02"],
        example_id="ex-1",
        dataset_name="benchmark_v1",
    )
    assert gt.query == "What is hybrid search?"
    assert gt.relevant_chunk_ids == ["chunk-01", "chunk-02"]
    assert gt.example_id == "ex-1"


def test_ground_truth_empty_relevant_set():
    """Test GroundTruth with empty relevant chunk set is permitted."""
    gt = GroundTruth(query="Query with no known chunks", relevant_chunk_ids=[])
    assert gt.relevant_chunk_ids == []


def test_ground_truth_validation():
    """Test GroundTruth validation on empty query and deduplication."""
    with pytest.raises(ValueError, match="Evaluation query cannot be empty"):
        GroundTruth(query="   ", relevant_chunk_ids=["chunk-1"])

    # Deduplicates and strips chunk IDs
    gt = GroundTruth(
        query="Valid query",
        relevant_chunk_ids=[" chunk-1 ", "chunk-2", "chunk-1"],
    )
    assert gt.relevant_chunk_ids == ["chunk-1", "chunk-2"]


# ============================================================================
# 2. RECALL@K TESTS
# ============================================================================


def test_recall_at_k_perfect():
    """Test perfect Recall@K where all relevant chunks appear in top K."""
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = ["c1", "c2"]
    assert calculate_recall_at_k(retrieved, relevant, k=2) == 1.0
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0


def test_recall_at_k_partial():
    """Test partial Recall@K where some relevant chunks appear in top K."""
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1", "c4", "c5"]  # 3 relevant, 1 retrieved
    assert pytest.approx(calculate_recall_at_k(retrieved, relevant, k=3), rel=1e-3) == 1 / 3


def test_recall_at_k_zero():
    """Test zero Recall@K when no relevant chunks are retrieved in top K."""
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c9", "c10"]
    assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.0


def test_recall_at_k_cutoff_boundary():
    """Test relevant chunk appearing after cutoff K is not counted."""
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = ["c4"]  # c4 is at rank 4
    assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0


def test_recall_at_k_fewer_results_than_k():
    """Test Recall@K when fewer results than K are retrieved."""
    retrieved = ["c1", "c2"]
    relevant = ["c1", "c3"]  # 2 relevant, 1 retrieved
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.5


def test_recall_at_k_empty_cases():
    """Test Recall@K with empty relevant set or empty retrieved results."""
    assert calculate_recall_at_k([], ["c1"], k=5) == 0.0
    assert calculate_recall_at_k(["c1"], [], k=5) == 0.0
    assert calculate_recall_at_k([], [], k=5) == 0.0


# ============================================================================
# 3. PRECISION@K TESTS
# ============================================================================


def test_precision_at_k_perfect():
    """Test perfect Precision@K where all items in top K are relevant."""
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1", "c2", "c3", "c4"]
    assert calculate_precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_partial():
    """Test partial Precision@K."""
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = ["c1", "c3"]  # 2 out of 4 relevant in top 4
    assert calculate_precision_at_k(retrieved, relevant, k=4) == 0.5


def test_precision_at_k_zero():
    """Test zero Precision@K when no top K items are relevant."""
    retrieved = ["c1", "c2"]
    relevant = ["c9", "c10"]
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.0


def test_precision_at_k_fewer_than_k():
    """Test Precision@K uses actual retrieved count in denominator if fewer than K."""
    retrieved = ["c1", "c2"]
    relevant = ["c1"]  # 1 relevant out of 2 retrieved for K=5
    # Denominator is min(5, 2) = 2 -> 1 / 2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.5


def test_precision_at_k_empty_cases():
    """Test Precision@K with empty retrieved or empty relevant."""
    assert calculate_precision_at_k([], ["c1"], k=5) == 0.0
    assert calculate_precision_at_k(["c1"], [], k=5) == 0.0


def test_precision_at_k_duplicate_ids():
    """Test Precision@K handles duplicate chunk IDs in retrieved results safely."""
    retrieved = ["c1", "c1", "c2"]
    relevant = ["c1"]
    # Top 2 has unique hit 'c1' out of 2 slots -> 1 / 2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5


# ============================================================================
# 4. MRR / RECIPROCAL RANK TESTS
# ============================================================================


def test_reciprocal_rank_first_hit():
    """Test Reciprocal Rank when first result is relevant (rank 1 -> RR = 1.0)."""
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1"]
    rr, rank = calculate_reciprocal_rank(retrieved, relevant)
    assert rr == 1.0
    assert rank == 1


def test_reciprocal_rank_later_hit():
    """Test Reciprocal Rank when first relevant result is at rank 2 or 3."""
    retrieved = ["c0", "c1", "c2"]
    relevant = ["c1"]
    rr, rank = calculate_reciprocal_rank(retrieved, relevant)
    assert rr == 0.5
    assert rank == 2

    retrieved2 = ["c0", "c9", "c1"]
    rr2, rank2 = calculate_reciprocal_rank(retrieved2, relevant)
    assert pytest.approx(rr2, rel=1e-3) == 1 / 3
    assert rank2 == 3


def test_reciprocal_rank_no_relevant():
    """Test Reciprocal Rank when no retrieved result is relevant (RR = 0.0)."""
    retrieved = ["c1", "c2"]
    relevant = ["c99"]
    rr, rank = calculate_reciprocal_rank(retrieved, relevant)
    assert rr == 0.0
    assert rank is None


def test_reciprocal_rank_multiple_relevant():
    """Test Reciprocal Rank takes the rank of the FIRST relevant hit."""
    retrieved = ["c0", "c1", "c2", "c3"]
    relevant = ["c2", "c1"]  # c1 is seen first at rank 2
    rr, rank = calculate_reciprocal_rank(retrieved, relevant)
    assert rr == 0.5
    assert rank == 2


def test_reciprocal_rank_empty():
    """Test Reciprocal Rank on empty retrieved or empty relevant set."""
    assert calculate_reciprocal_rank([], ["c1"]) == (0.0, None)
    assert calculate_reciprocal_rank(["c1"], []) == (0.0, None)


# ============================================================================
# 5. GROUNDEDNESS / CONTEXT ASSERTION OVERLAP TESTS
# ============================================================================


def test_context_assertion_overlap_full():
    """Test 100% lexical overlap between assertion and context."""
    assertion = "Paris is the capital of France."
    context = "Paris is the capital of France and has a large population."
    res = calculate_context_assertion_overlap(assertion, context)
    assert res.context_assertion_overlap == 1.0
    assert res.overlap_token_count == len(res.total_assertion_tokens * [1])


def test_context_assertion_overlap_partial():
    """Test partial lexical overlap."""
    assertion = "Paris is the capital and largest city."
    context = "Paris is the capital."
    res = calculate_context_assertion_overlap(assertion, context)
    # Assertion tokens: paris, is, the, capital, and, largest, city (7 tokens)
    # Context tokens: paris, is, the, capital (4 tokens)
    # Overlap: 4 / 7 = ~0.5714
    assert 0.5 < res.context_assertion_overlap < 0.6
    assert res.overlap_token_count == 4


def test_context_assertion_overlap_zero():
    """Test zero lexical overlap."""
    assertion = "Quantum computing uses qubits."
    context = "Ancient Roman history and philosophy."
    res = calculate_context_assertion_overlap(assertion, context)
    assert res.context_assertion_overlap == 0.0
    assert res.overlap_token_count == 0


def test_context_assertion_overlap_normalization():
    """Test punctuation and casing do not prevent lexical matching."""
    assertion = "HELLO, WORLD! (TEST-123)."
    context = "hello world test 123"
    res = calculate_context_assertion_overlap(assertion, context)
    assert res.context_assertion_overlap == 1.0


def test_context_assertion_overlap_empty_inputs():
    """Test empty assertion or empty context produces 0.0 safely."""
    assert calculate_context_assertion_overlap("", "some context").context_assertion_overlap == 0.0
    assert (
        calculate_context_assertion_overlap("some assertion", "").context_assertion_overlap == 0.0
    )
    assert calculate_context_assertion_overlap("", "").context_assertion_overlap == 0.0


# ============================================================================
# 6. AGGREGATE METRICS TESTS
# ============================================================================


def test_aggregate_retrieval_metrics():
    """Test mean aggregation across multiple retrieval metrics."""
    m1 = calculate_retrieval_metrics(["c1"], ["c1"], k=1)  # recall=1, prec=1, mrr=1
    m2 = calculate_retrieval_metrics(["c0"], ["c1"], k=1)  # recall=0, prec=0, mrr=0
    mean_rec, mean_prec, mean_mrr = aggregate_retrieval_metrics([m1, m2])
    assert mean_rec == 0.5
    assert mean_prec == 0.5
    assert mean_mrr == 0.5


def test_aggregate_retrieval_metrics_empty():
    """Test aggregating empty metrics list."""
    assert aggregate_retrieval_metrics([]) == (0.0, 0.0, 0.0)


# ============================================================================
# 7. LLM JUDGE INTERFACE & MOCK TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_base_llm_judge_interface():
    """Test BaseLLMJudge interface contract."""

    class CustomJudge(BaseLLMJudge):
        async def evaluate(self, query, answer, context=None, criteria=None):
            return JudgeEvaluationResult(score=0.9, passed=True, reasoning="Custom passed")

    judge = CustomJudge()
    res = await judge.evaluate("query", "answer")
    assert res.score == 0.9
    assert res.passed is True


@pytest.mark.asyncio
async def test_mock_llm_judge():
    """Test deterministic MockLLMJudge without any live API calls."""
    mock_judge = MockLLMJudge(default_score=0.85, default_passed=True)
    res = await mock_judge.evaluate(
        query="What is RAG?",
        answer="Retrieval-Augmented Generation.",
        context="RAG stands for Retrieval-Augmented Generation.",
    )
    assert res.score == 0.85
    assert res.passed is True
    assert len(mock_judge.call_history) == 1


@pytest.mark.asyncio
async def test_llm_judge_with_mock_provider():
    """Test LLMJudge structured evaluation using deterministic MockLLMProvider."""
    mock_provider = MockLLMProvider()
    mock_provider.queue_response(
        LLMResponse(
            content=(
                '{"score": 0.95, "passed": true, '
                '"reasoning": "Answer is fully grounded in the retrieved context."}'
            )
        )
    )

    judge = LLMJudge(provider=mock_provider)
    res = await judge.evaluate(
        query="What is LangGraph?",
        answer="LangGraph coordinates multi-agent graphs.",
        context="LangGraph coordinates multi-agent graphs.",
    )

    assert res.score == 0.95
    assert res.passed is True
    assert "fully grounded" in res.reasoning


@pytest.mark.asyncio
async def test_llm_judge_malformed_response_handling():
    """Test LLMJudge safely handles non-JSON output without raising an exception."""
    mock_provider = MockLLMProvider()
    mock_provider.queue_response(LLMResponse(content="I cannot evaluate this request properly."))

    judge = LLMJudge(provider=mock_provider)
    res = await judge.evaluate(query="Q", answer="A")
    assert res.score == 0.0
    assert res.passed is False
    assert "Failed to parse" in res.reasoning


# ============================================================================
# 8. BENCHMARK FIXTURES & DATASET EVALUATION
# ============================================================================


def test_deterministic_benchmark_cases():
    """Verify all 10 benchmark cases are structured with explicit ground-truth chunks."""
    cases = create_deterministic_benchmark_cases()
    assert len(cases) == 10

    case_ids = [c.case_id for c in cases]
    assert "case-rank-1" in case_ids
    assert "case-rank-3" in case_ids
    assert "case-no-relevant" in case_ids
    assert "case-multiple-relevant" in case_ids
    assert "case-fewer-than-k" in case_ids
    assert "case-empty-relevant" in case_ids
    assert "case-empty-retrieved" in case_ids
    assert "case-grounded-overlap" in case_ids


@pytest.mark.asyncio
async def test_rag_evaluator_evaluate_dataset():
    """Test running RAGEvaluator across the deterministic benchmark dataset."""
    dataset = get_deterministic_benchmark_dataset()
    evaluator = RAGEvaluator(default_k=5, judge=MockLLMJudge(default_score=1.0))

    result = await evaluator.evaluate_dataset(dataset)

    assert result.total_cases == 10
    assert 0.0 <= result.mean_recall_at_k <= 1.0
    assert 0.0 <= result.mean_precision_at_k <= 1.0
    assert 0.0 <= result.mean_mrr <= 1.0
    assert result.mean_context_assertion_overlap is not None
    assert result.mean_judge_score == 1.0
    assert len(result.case_results) == 10


# ============================================================================
# 9. INTEGRATION WITH RAGSERVICE
# ============================================================================


@pytest.mark.asyncio
async def test_evaluator_live_rag_service_integration():
    """Test RAGEvaluator evaluating live against an in-memory RAGService."""
    # 1. Setup isolated in-memory Chroma + mock embeddings
    settings = Settings(rag_enabled=True, rag_embedding_provider="mock")
    store = ChromaVectorStore(collection_name="test_eval_live_kb", ephemeral=True)
    embedder = MockEmbeddingProvider(dimension=16)
    rag_service = RAGService(vector_store=store, embedding_provider=embedder, settings=settings)

    # 2. Ingest two documents
    _ = await rag_service.ingest_text(
        content="Antigravity is an agentic AI coding platform built for robust orchestration.",
        filename="antigravity_intro.txt",
    )
    doc2 = await rag_service.ingest_text(
        content="Hybrid search merges dense semantic vectors with sparse BM25 indexing.",
        filename="hybrid_search.txt",
    )

    # Find chunk ID of the first chunk of doc2
    hybrid_chunk_id = None
    chunks = await store.get_all_chunks(filter_metadata={"document_id": doc2.id})
    if chunks:
        hybrid_chunk_id = chunks[0].id

    assert hybrid_chunk_id is not None

    # 3. Create evaluation case expecting hybrid_chunk_id
    case = EvaluationCase(
        query="hybrid search BM25",
        relevant_chunk_ids=[hybrid_chunk_id],
        generated_answer="Hybrid search uses BM25 and semantic vectors.",
    )

    evaluator = RAGEvaluator(default_k=5)
    case_result = await evaluator.evaluate_case(case=case, rag_service=rag_service, k=5)

    assert case_result.retrieval_metrics.relevant_count == 1
    # Relevant chunk should be retrieved in top 5
    assert case_result.retrieval_metrics.recall_at_k == 1.0
    assert case_result.groundedness is not None
    assert case_result.groundedness.context_assertion_overlap > 0.0


# ============================================================================
# 10. OBSERVABILITY & CARDINALITY TEST
# ============================================================================


@pytest.mark.asyncio
async def test_evaluation_telemetry_low_cardinality():
    """Verify evaluation spans record without high-cardinality attributes."""
    from src.app.observability import init_telemetry

    init_telemetry(
        Settings(
            telemetry_enabled=True,
            telemetry_exporter="memory",
            telemetry_service_name="eval-test",
        )
    )
    async with trace_span("evaluation.run", attributes={"dataset": "test", "k": 5}) as span:
        assert span.is_recording()

    # Verify no prohibited attributes appear in models or result schemas
    result = calculate_retrieval_metrics(["c1"], ["c1"], k=5)
    result_dict = result.model_dump()

    prohibited_keys = {
        "query",
        "task_id",
        "thread_id",
        "scope_id",
        "trace_id",
        "span_id",
        "prompt",
        "content",
    }
    for key in result_dict.keys():
        assert key not in prohibited_keys
