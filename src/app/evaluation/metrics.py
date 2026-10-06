"""Deterministic retrieval and groundedness metrics for RAG evaluation.

Mathematical Definitions:
1. Recall@K:
   Proportion of ground-truth relevant chunks retrieved within the top K positions.
   Recall@K = |Unique relevant chunks in top K| / |Total ground-truth relevant chunks|
   - If total ground-truth chunks == 0: 0.0
   - If retrieved chunks == 0: 0.0

2. Precision@K:
   Proportion of retrieved items in the top K positions that are relevant.
   Precision@K = |Unique relevant chunks in top K| / min(K, len(retrieved_chunks))
   - If fewer than K chunks are retrieved, the denominator is the actual count retrieved.
   - If retrieved chunks == 0: 0.0

3. Reciprocal Rank (MRR):
   Reciprocal of the 1-based rank of the FIRST ground-truth relevant chunk in retrieved results.
   RR = 1.0 / rank (or 0.0 if no relevant chunk is retrieved)

4. Context Assertion Overlap (context_assertion_overlap):
   Deterministic lexical proxy measuring the token overlap between an assertion and context.
   NOTE: This is a lexical heuristic proxy only, NOT a semantic entailment or factuality evaluator.
"""

import re
from collections import Counter
from typing import List, Optional, Sequence, Set, Tuple

from src.app.evaluation.models import GroundednessResult, RetrievalMetrics


def calculate_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate Recall@K: fraction of relevant chunks retrieved in top K.

    Args:
        retrieved_chunk_ids: Ordered sequence of retrieved chunk IDs.
        relevant_chunk_ids: Ground-truth relevant chunk IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        Float score bounded in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    relevant_set: Set[str] = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    if not retrieved_chunk_ids:
        return 0.0

    top_k_retrieved: List[str] = list(retrieved_chunk_ids)[:k]
    hits = len(set(top_k_retrieved) & relevant_set)
    return float(hits / len(relevant_set))


def calculate_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate Precision@K: fraction of top K retrieved chunks that are relevant.

    If fewer than K results are retrieved, uses the actual number of retrieved results
    in the denominator.

    Args:
        retrieved_chunk_ids: Ordered sequence of retrieved chunk IDs.
        relevant_chunk_ids: Ground-truth relevant chunk IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        Float score bounded in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if not retrieved_chunk_ids:
        return 0.0

    relevant_set: Set[str] = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    top_k_retrieved: List[str] = list(retrieved_chunk_ids)[:k]
    hits = len(set(top_k_retrieved) & relevant_set)

    # Use actual retrieved count if fewer than K results
    denominator = min(k, len(retrieved_chunk_ids))
    if denominator <= 0:
        return 0.0

    return float(hits / denominator)


def calculate_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> Tuple[float, Optional[int]]:
    """Calculate Reciprocal Rank (RR) for the first relevant retrieved chunk.

    Args:
        retrieved_chunk_ids: Ordered sequence of retrieved chunk IDs.
        relevant_chunk_ids: Ground-truth relevant chunk IDs.

    Returns:
        Tuple of (reciprocal_rank, 1_based_rank_of_first_hit).
        If no relevant result was retrieved: (0.0, None).
    """
    relevant_set: Set[str] = set(relevant_chunk_ids)
    if not relevant_set or not retrieved_chunk_ids:
        return 0.0, None

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            return float(1.0 / rank), rank

    return 0.0, None


def calculate_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Recall@K, Precision@K, and Reciprocal Rank for a retrieval outcome."""
    recall = calculate_recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    precision = calculate_precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    rr, first_rank = calculate_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids)

    relevant_set = set(relevant_chunk_ids)
    top_k_slice = list(retrieved_chunk_ids)[:k]
    relevant_in_k = len(set(top_k_slice) & relevant_set)

    return RetrievalMetrics(
        recall_at_k=round(recall, 4),
        precision_at_k=round(precision, 4),
        mrr=round(rr, 4),
        k=k,
        relevant_count=len(relevant_set),
        retrieved_count=len(retrieved_chunk_ids),
        relevant_retrieved_in_k=relevant_in_k,
        first_relevant_rank=first_rank,
    )


def calculate_context_assertion_overlap(
    assertion: str,
    context: str,
) -> GroundednessResult:
    """Compute deterministic lexical token overlap between assertion and retrieved context.

    CRITICAL NOTE:
    This is a deterministic lexical proxy metric. It measures multiset token containment
    after case- and punctuation-normalization. It does NOT perform semantic entailment,
    natural language inference, or factuality verification.

    Args:
        assertion: The statement or generated text to evaluate.
        context: The retrieved background text/context.

    Returns:
        GroundednessResult containing overlap score in [0.0, 1.0] and token statistics.
    """
    if not assertion or not assertion.strip():
        return GroundednessResult(
            context_assertion_overlap=0.0,
            overlap_token_count=0,
            total_assertion_tokens=0,
            total_context_tokens=len(re.findall(r"\b[a-zA-Z0-9]+\b", context.lower()))
            if context
            else 0,
            overlap_tokens=[],
        )

    if not context or not context.strip():
        assertion_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", assertion.lower())
        return GroundednessResult(
            context_assertion_overlap=0.0,
            overlap_token_count=0,
            total_assertion_tokens=len(assertion_tokens),
            total_context_tokens=0,
            overlap_tokens=[],
        )

    assertion_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", assertion.lower())
    context_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", context.lower())

    if not assertion_tokens:
        return GroundednessResult(
            context_assertion_overlap=0.0,
            overlap_token_count=0,
            total_assertion_tokens=0,
            total_context_tokens=len(context_tokens),
            overlap_tokens=[],
        )

    a_counter = Counter(assertion_tokens)
    c_counter = Counter(context_tokens)

    overlap_count = 0
    common_tokens: List[str] = []
    for token, a_count in a_counter.items():
        c_count = c_counter.get(token, 0)
        matched = min(a_count, c_count)
        overlap_count += matched
        if matched > 0:
            common_tokens.append(token)

    score = float(overlap_count / len(assertion_tokens))
    bounded_score = max(0.0, min(1.0, score))

    return GroundednessResult(
        context_assertion_overlap=round(bounded_score, 4),
        overlap_token_count=overlap_count,
        total_assertion_tokens=len(assertion_tokens),
        total_context_tokens=len(context_tokens),
        overlap_tokens=sorted(common_tokens),
    )


def aggregate_retrieval_metrics(
    metrics_list: Sequence[RetrievalMetrics],
) -> Tuple[float, float, float]:
    """Calculate mean Recall@K, mean Precision@K, and Mean Reciprocal Rank (MRR)."""
    if not metrics_list:
        return 0.0, 0.0, 0.0

    n = len(metrics_list)
    mean_recall = sum(m.recall_at_k for m in metrics_list) / n
    mean_precision = sum(m.precision_at_k for m in metrics_list) / n
    mean_mrr = sum(m.mrr for m in metrics_list) / n

    return round(mean_recall, 4), round(mean_precision, 4), round(mean_mrr, 4)
