"""Hybrid retrieval engine combining dense semantic search and BM25 lexical ranking."""

import math
import re
from typing import Any, Dict, List, Optional, Set

from src.app.core.logging import get_logger
from src.app.rag.embeddings import EmbeddingProvider
from src.app.rag.models import DocumentChunk, RetrievalResult
from src.app.rag.vector_store import VectorStore

logger = get_logger(__name__)


class BM25Scorer:
    """Okapi BM25 implementation for lexical keyword search across document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.stop_words: Set[str] = {
            "a",
            "about",
            "above",
            "after",
            "again",
            "against",
            "all",
            "am",
            "an",
            "and",
            "any",
            "are",
            "aren't",
            "as",
            "at",
            "be",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "but",
            "by",
            "can",
            "can't",
            "cannot",
            "could",
            "couldn't",
            "did",
            "didn't",
            "do",
            "does",
            "doesn't",
            "doing",
            "don't",
            "down",
            "during",
            "each",
            "few",
            "for",
            "from",
            "further",
            "had",
            "hadn't",
            "has",
            "hasn't",
            "have",
            "haven't",
            "having",
            "he",
            "her",
            "here",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "i",
            "if",
            "in",
            "into",
            "is",
            "isn't",
            "it",
            "its",
            "itself",
            "let's",
            "me",
            "more",
            "most",
            "mustn't",
            "my",
            "myself",
            "no",
            "nor",
            "not",
            "of",
            "off",
            "on",
            "once",
            "only",
            "or",
            "other",
            "ought",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "own",
            "same",
            "shan't",
            "she",
            "should",
            "shouldn't",
            "so",
            "some",
            "such",
            "than",
            "that",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "there",
            "these",
            "they",
            "this",
            "those",
            "through",
            "to",
            "too",
            "under",
            "until",
            "up",
            "very",
            "was",
            "wasn't",
            "we",
            "were",
            "weren't",
            "what",
            "when",
            "where",
            "which",
            "while",
            "who",
            "whom",
            "why",
            "with",
            "won't",
            "would",
            "wouldn't",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
        }

    def tokenize(self, text: str) -> List[str]:
        """Convert text into normalized tokens."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if len(t) > 1 and t not in self.stop_words]

    def score(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Compute BM25 scores for all chunks and return top_k ranked results."""
        if not chunks or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            query_tokens = [w.lower() for w in query.split() if w.strip()]
            if not query_tokens:
                return []

        num_docs = len(chunks)
        doc_tokens = [self.tokenize(c.content) for c in chunks]
        doc_lens = [len(tokens) for tokens in doc_tokens]
        avg_doc_len = sum(doc_lens) / max(1, num_docs)

        df: Dict[str, int] = {}
        for q in query_tokens:
            df[q] = sum(1 for tokens in doc_tokens if q in tokens)

        scores: List[float] = []
        for i, chunk in enumerate(chunks):
            tokens = doc_tokens[i]
            doc_len = doc_lens[i]
            chunk_score = 0.0

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for q in query_tokens:
                q_df = df.get(q, 0)
                if q_df == 0:
                    continue

                idf = math.log((num_docs - q_df + 0.5) / (q_df + 0.5) + 1.0)
                q_tf = tf.get(q, 0)

                numerator = q_tf * (self.k1 + 1.0)
                denominator = q_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                chunk_score += idf * (numerator / max(0.001, denominator))

            scores.append(chunk_score)

        scored_pairs = [(chunks[i], scores[i]) for i in range(num_docs) if scores[i] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, s in scored_pairs[:top_k]:
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=round(s, 4),
                    source_type="bm25",
                )
            )

        return results


class HybridRetriever:
    """Hybrid retriever combining dense semantic search and BM25 with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        bm25_scorer: Optional[BM25Scorer] = None,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.6,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_scorer = bm25_scorer or BM25Scorer()
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.alpha = alpha

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Execute hybrid search using the requested or default fusion strategy."""
        selected_strategy = (strategy or self.fusion_strategy).lower()

        # Step 1: Dense semantic vector search
        query_embedding = await self.embedding_provider.embed_query(query)
        candidate_k = max(top_k * 2, 10)
        semantic_results = await self.vector_store.search_vector(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filter_metadata=filter_metadata,
        )

        if selected_strategy == "semantic":
            return semantic_results[:top_k]

        # Step 2: Lexical BM25 search over candidate chunks
        all_chunks = await self.vector_store.get_all_chunks(filter_metadata=filter_metadata)
        bm25_results = self.bm25_scorer.score(
            query=query,
            chunks=all_chunks,
            top_k=candidate_k,
        )

        if selected_strategy == "bm25":
            return bm25_results[:top_k]

        # Step 3: Fusion
        if selected_strategy == "weighted":
            return self._linear_weighted_fusion(semantic_results, bm25_results, top_k)
        else:
            return self._reciprocal_rank_fusion(semantic_results, bm25_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Fuse two result sets using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        for rank, res in enumerate(semantic_results, start=1):
            chunk_map[res.chunk_id] = res
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, res in enumerate(bm25_results, start=1):
            chunk_map[res.chunk_id] = res
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_ids[:top_k]:
            base_res = chunk_map[cid]
            final_results.append(
                RetrievalResult(
                    chunk_id=base_res.chunk_id,
                    document_id=base_res.document_id,
                    content=base_res.content,
                    metadata=base_res.metadata,
                    score=round(rrf_scores[cid], 5),
                    source_type="hybrid_rrf",
                )
            )

        return final_results

    def _linear_weighted_fusion(
        self,
        semantic_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Fuse two result sets using min-max normalized linear weighted combination."""
        chunk_map: Dict[str, RetrievalResult] = {}
        sem_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for res in semantic_results:
            chunk_map[res.chunk_id] = res
            sem_scores[res.chunk_id] = res.score

        for res in bm25_results:
            chunk_map[res.chunk_id] = res
            bm25_scores[res.chunk_id] = res.score

        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores.values()) if bm25_scores else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

        all_ids = set(sem_scores.keys()).union(bm25_scores.keys())
        combined_scores: Dict[str, float] = {}

        for cid in all_ids:
            s_score = sem_scores.get(cid, 0.0)
            b_raw = bm25_scores.get(cid, 0.0)
            b_norm = (b_raw - min_bm25) / bm25_range if b_raw > 0 else 0.0

            combined = (self.alpha * s_score) + ((1.0 - self.alpha) * b_norm)
            combined_scores[cid] = combined

        sorted_ids = sorted(all_ids, key=lambda cid: combined_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_ids[:top_k]:
            base_res = chunk_map[cid]
            final_results.append(
                RetrievalResult(
                    chunk_id=base_res.chunk_id,
                    document_id=base_res.document_id,
                    content=base_res.content,
                    metadata=base_res.metadata,
                    score=round(combined_scores[cid], 4),
                    source_type="hybrid_weighted",
                )
            )

        return final_results
