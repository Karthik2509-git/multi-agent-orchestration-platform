# ADR 003: Hybrid Retrieval-Augmented Generation (RAG) Architecture

## Status
Accepted

## Context
Retrieval-Augmented Generation systems relying solely on dense vector search (embeddings) often fail when queries contain exact product names, error codes, unique identifiers, or technical acronyms. Conversely, purely lexical search engines (like BM25) fail when queries use synonyms, paraphrased expressions, or high-level concepts not verbatim in the text. Furthermore, relying on remote embedding APIs introduces network latency, variable costs, and offline development obstacles.

## Decision
Implement a **Hybrid RAG architecture** combining local dense vector search and sparse lexical search, fused using **Reciprocal Rank Fusion (RRF)**:
- **Dense Vector Search**: Chroma vector database utilizing local ONNX embeddings (`all-MiniLM-L6-v2`), with an in-memory mock provider for deterministic tests.
- **Sparse Lexical Search**: BM25 keyword index tokenizing terms and evaluating term frequency / inverse document frequency.
- **Reciprocal Rank Fusion (RRF)**: Merges the ranked lists from both engines using the formula:
  $$RRF(d) = \alpha \cdot \frac{1}{k + r_{\text{dense}}(d)} + (1 - \alpha) \cdot \frac{1}{k + r_{\text{lexical}}(d)}$$
  Configured with dense weight $\alpha = 0.6$ and rank constant $k = 60$.
- **Ingestion Pipeline**: Multi-format document parser (`.txt`, `.md`, `.pdf`) coupled with recursive character chunking (500 chars, 50 overlap).

## Consequences

### Positive
- **High Retrieval Precision**: Combines semantic generalization with exact keyword recall.
- **Zero API Cost & 100% Offline Capability**: Local ONNX inference requires no paid API keys or external network requests.
- **No Heavy Reranker Overhead**: RRF is an efficient rank-merging algorithm that avoids the high latency of running a secondary cross-encoder transformer model.

### Negative / Trade-Offs
- In-memory BM25 index requires synchronization with the Chroma vector store during document deletions or updates.
- ONNX runtime dependency adds initial model download and slight memory overhead on startup.

## Architectural Comparison
- **Vector-Only Retrieval**: Simpler implementation, but poor performance on exact phrase matching, code snippets, and identifiers.
- **Cross-Encoder Reranking (e.g. Cohere / BGE-Reranker)**: Produces marginal precision gains over RRF, but adds significant computational latency and external dependency overhead.
