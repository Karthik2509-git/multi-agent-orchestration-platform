"""Domain models and data structures for the Evaluation Framework (Phase 7 Milestone 4)."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.app.rag.models import RetrievalResult


class GroundTruth(BaseModel):
    """Ground truth relevance annotation identifying relevant chunks for a query."""

    query: str
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    example_id: Optional[str] = None
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Evaluation query cannot be empty.")
        return trimmed

    @field_validator("relevant_chunk_ids", mode="before")
    @classmethod
    def validate_relevant_chunk_ids(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            cleaned: List[str] = []
            seen = set()
            for item in v:
                s = str(item).strip()
                if s and s not in seen:
                    cleaned.append(s)
                    seen.add(s)
            return cleaned
        raise ValueError("relevant_chunk_ids must be a collection of chunk ID strings.")


class EvaluationCase(BaseModel):
    """A single evaluation case containing query, ground truth, and optional candidate outputs."""

    case_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    retrieved_results: Optional[List[RetrievalResult]] = None
    retrieved_chunk_ids: Optional[List[str]] = None
    reference_answer: Optional[str] = None
    generated_answer: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Evaluation query cannot be empty.")
        return trimmed

    def get_retrieved_chunk_ids(self) -> List[str]:
        """Extract ordered retrieved chunk IDs from explicit IDs or RetrievalResults."""
        if self.retrieved_chunk_ids is not None:
            return list(self.retrieved_chunk_ids)
        if self.retrieved_results is not None:
            return [res.chunk_id for res in self.retrieved_results]
        return []

    def get_context_text(self) -> str:
        """Return explicit context text or concatenated content from retrieved results."""
        if self.context:
            return self.context
        if self.retrieved_results:
            return "\n\n".join(r.content for r in self.retrieved_results if r.content)
        return ""


class EvaluationDataset(BaseModel):
    """Collection of evaluation cases defining a benchmark suite."""

    name: str
    cases: List[EvaluationCase] = Field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    """Deterministic retrieval quality metrics for a single query."""

    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    k: int = Field(gt=0)
    relevant_count: int = Field(ge=0)
    retrieved_count: int = Field(ge=0)
    relevant_retrieved_in_k: int = Field(ge=0)
    first_relevant_rank: Optional[int] = None


class GroundednessResult(BaseModel):
    """Deterministic lexical overlap groundedness metric proxy."""

    context_assertion_overlap: float = Field(ge=0.0, le=1.0)
    overlap_token_count: int = Field(ge=0)
    total_assertion_tokens: int = Field(ge=0)
    total_context_tokens: int = Field(ge=0)
    overlap_tokens: List[str] = Field(default_factory=list)


class JudgeEvaluationResult(BaseModel):
    """Structured evaluation output produced by an optional semantic LLM judge."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasoning: str
    criteria: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationCaseResult(BaseModel):
    """Complete evaluation output for an individual evaluation case."""

    case_id: str
    query: str
    retrieval_metrics: RetrievalMetrics
    groundedness: Optional[GroundednessResult] = None
    judge_result: Optional[JudgeEvaluationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AggregateEvaluationResult(BaseModel):
    """Aggregated retrieval and groundedness benchmark metrics across multiple cases."""

    dataset_name: Optional[str] = None
    total_cases: int = Field(ge=0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_mrr: float = Field(ge=0.0, le=1.0)
    mean_context_assertion_overlap: Optional[float] = None
    mean_judge_score: Optional[float] = None
    k: int = Field(gt=0)
    case_results: List[EvaluationCaseResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
