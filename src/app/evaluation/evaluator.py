"""RAG Evaluation orchestrator computing retrieval and groundedness metrics."""

from typing import List, Optional

from src.app.core.logging import get_logger
from src.app.evaluation.judge import BaseLLMJudge
from src.app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    calculate_context_assertion_overlap,
    calculate_retrieval_metrics,
)
from src.app.evaluation.models import (
    AggregateEvaluationResult,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    GroundednessResult,
    JudgeEvaluationResult,
)
from src.app.observability import set_span_attributes, trace_span
from src.app.rag.service import RAGService

logger = get_logger(__name__)


class RAGEvaluator:
    """Evaluates RAG retrieval and groundedness deterministically with optional judge scoring."""

    def __init__(
        self,
        default_k: int = 5,
        judge: Optional[BaseLLMJudge] = None,
    ):
        if default_k <= 0:
            raise ValueError("default_k must be a positive integer.")
        self.default_k = default_k
        self.judge = judge

    async def evaluate_case(
        self,
        case: EvaluationCase,
        rag_service: Optional[RAGService] = None,
        k: Optional[int] = None,
    ) -> EvaluationCaseResult:
        """Evaluate a single evaluation case.

        If a RAGService is provided, retrieves live results for case.query.
        Otherwise, evaluates candidate results already stored on the EvaluationCase.

        Args:
            case: The EvaluationCase to evaluate.
            rag_service: Optional live RAGService instance.
            k: Cutoff rank (defaults to evaluator default_k).

        Returns:
            EvaluationCaseResult with retrieval metrics and optional groundedness/judge results.
        """
        effective_k = k or self.default_k

        # 1. Retrieve candidates if rag_service provided, else use static case candidates
        if rag_service is not None:
            results = await rag_service.retrieve(query=case.query, top_k=effective_k)
            retrieved_chunk_ids = [r.chunk_id for r in results]
            context_text = "\n\n".join(r.content for r in results if r.content)
        else:
            retrieved_chunk_ids = case.get_retrieved_chunk_ids()
            context_text = case.get_context_text()

        # 2. Compute deterministic retrieval metrics
        retrieval_metrics = calculate_retrieval_metrics(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=case.relevant_chunk_ids,
            k=effective_k,
        )

        # 3. Compute deterministic lexical groundedness if generated answer is available
        groundedness: Optional[GroundednessResult] = None
        if case.generated_answer is not None:
            groundedness = calculate_context_assertion_overlap(
                assertion=case.generated_answer,
                context=context_text,
            )

        # 4. Optional semantic LLM judge evaluation
        judge_result: Optional[JudgeEvaluationResult] = None
        if self.judge is not None and case.generated_answer is not None:
            criteria = case.metadata.get("criteria")
            judge_result = await self.judge.evaluate(
                query=case.query,
                answer=case.generated_answer,
                context=context_text or None,
                criteria=criteria,
            )

        return EvaluationCaseResult(
            case_id=case.case_id,
            query=case.query,
            retrieval_metrics=retrieval_metrics,
            groundedness=groundedness,
            judge_result=judge_result,
            metadata=dict(case.metadata),
        )

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        rag_service: Optional[RAGService] = None,
        k: Optional[int] = None,
    ) -> AggregateEvaluationResult:
        """Evaluate an entire dataset and compute aggregate retrieval and groundedness scores.

        Emits an OpenTelemetry span with low-cardinality attributes.
        """
        effective_k = k or self.default_k

        # Strictly low-cardinality span attributes (no query text, task IDs, or chunks)
        span_attrs = {
            "dataset_name": dataset.name,
            "total_cases": len(dataset.cases),
            "k": effective_k,
            "has_judge": self.judge is not None,
        }

        async with trace_span("evaluation.run", attributes=span_attrs) as span:
            case_results: List[EvaluationCaseResult] = []
            for case in dataset.cases:
                res = await self.evaluate_case(
                    case=case,
                    rag_service=rag_service,
                    k=effective_k,
                )
                case_results.append(res)

            # Aggregate retrieval metrics
            metrics_list = [cr.retrieval_metrics for cr in case_results]
            mean_recall, mean_precision, mean_mrr = aggregate_retrieval_metrics(metrics_list)

            # Aggregate groundedness if any cases had generated answers
            overlap_scores = [
                cr.groundedness.context_assertion_overlap
                for cr in case_results
                if cr.groundedness is not None
            ]
            mean_overlap = (
                round(sum(overlap_scores) / len(overlap_scores), 4) if overlap_scores else None
            )

            # Aggregate judge scores if any judge results
            judge_scores = [
                cr.judge_result.score for cr in case_results if cr.judge_result is not None
            ]
            mean_judge_score = (
                round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None
            )

            set_span_attributes(
                span,
                {
                    "mean_recall_at_k": mean_recall,
                    "mean_precision_at_k": mean_precision,
                    "mean_mrr": mean_mrr,
                    "status": "completed",
                },
            )

            return AggregateEvaluationResult(
                dataset_name=dataset.name,
                total_cases=len(case_results),
                mean_recall_at_k=mean_recall,
                mean_precision_at_k=mean_precision,
                mean_mrr=mean_mrr,
                mean_context_assertion_overlap=mean_overlap,
                mean_judge_score=mean_judge_score,
                k=effective_k,
                case_results=case_results,
                metadata={
                    "dataset_description": dataset.description,
                    **dataset.metadata,
                },
            )
