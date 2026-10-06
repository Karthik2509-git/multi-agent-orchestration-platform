"""Agent evaluation benchmarks, datasets, and metrics (Phase 7 Milestone 4)."""

from src.app.evaluation.evaluator import RAGEvaluator
from src.app.evaluation.fixtures import (
    create_deterministic_benchmark_cases,
    get_deterministic_benchmark_dataset,
)
from src.app.evaluation.judge import (
    BaseLLMJudge,
    LLMJudge,
    MockLLMJudge,
)
from src.app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    calculate_context_assertion_overlap,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_retrieval_metrics,
)
from src.app.evaluation.models import (
    AggregateEvaluationResult,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    GroundednessResult,
    GroundTruth,
    JudgeEvaluationResult,
    RetrievalMetrics,
)

__all__ = [
    "AggregateEvaluationResult",
    "BaseLLMJudge",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationDataset",
    "GroundednessResult",
    "GroundTruth",
    "JudgeEvaluationResult",
    "LLMJudge",
    "MockLLMJudge",
    "RAGEvaluator",
    "RetrievalMetrics",
    "aggregate_retrieval_metrics",
    "calculate_context_assertion_overlap",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
    "calculate_retrieval_metrics",
    "create_deterministic_benchmark_cases",
    "get_deterministic_benchmark_dataset",
]
