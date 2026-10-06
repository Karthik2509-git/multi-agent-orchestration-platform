#!/usr/bin/env python3
"""Deployment verification utility for Multi-Agent Orchestration Platform.

Purpose:
    Validates a RUNNING deployment through its HTTP API surface.
    Verifies Liveness, Readiness, RAG indexing/retrieval, Long-Term Semantic
    Memory create/search, and Orchestration/checkpoint execution using the
    active or Mock LLM provider.

IMPORTANT SCOPE NOTE:
    This script verifies that an actively running service is operational.
    It does NOT by itself prove container restart persistence across container
    recreations (which requires stopping and restarting the Docker daemon/containers).
"""

import argparse
import sys
import uuid
from typing import Any, Dict, Optional

import httpx


def _run_checks(client: httpx.Client) -> int:
    """Execute verification sequence using the provided HTTP client."""
    # Step A: Liveness Check
    try:
        resp = client.get("/health")
        if resp.status_code != 200:
            print(f"[FAIL] Liveness: Expected 200, got {resp.status_code}: {resp.text[:80]}")
            return 1
        data = resp.json()
        if data.get("status") != "healthy":
            print(f"[FAIL] Liveness: Unexpected status '{data.get('status')}'")
            return 1
        print("[PASS] Liveness")
    except Exception as e:
        print(f"[FAIL] Liveness: Connection error: {e}")
        return 1

    # Step B: Readiness Check
    try:
        resp = client.get("/api/v1/health")
        if resp.status_code != 200:
            print(f"[FAIL] Readiness: Expected 200, got {resp.status_code}: {resp.text[:80]}")
            return 1
        data = resp.json()
        if data.get("status") != "healthy":
            print(f"[FAIL] Readiness: Unexpected readiness status '{data.get('status')}'")
            return 1
        print("[PASS] Readiness")
    except Exception as e:
        print(f"[FAIL] Readiness: Connection error: {e}")
        return 1

    # Step C: RAG Functional Ingestion & Retrieval
    unique_token = f"deploy-token-{uuid.uuid4().hex[:8]}"
    try:
        ingest_payload = {
            "content": f"Platform deployment verification token: {unique_token}.",
            "filename": "deployment_verification.txt",
            "metadata": {"test_run": True},
        }
        resp = client.post("/api/v1/knowledge/documents", json=ingest_payload)
        if resp.status_code != 201:
            print(f"[FAIL] RAG ingestion: Expected 201, got {resp.status_code}")
            return 1

        # Query for the unique token
        search_payload = {
            "query": unique_token,
            "top_k": 3,
        }
        resp = client.post("/api/v1/knowledge/search", json=search_payload)
        if resp.status_code != 200:
            print(f"[FAIL] RAG retrieval: Expected 200, got {resp.status_code}")
            return 1
        search_data = resp.json()
        results = search_data.get("results", [])
        token_found = any(unique_token in r.get("content", "") for r in results)
        if not token_found:
            print(f"[FAIL] RAG retrieval: Token '{unique_token}' not found in search results.")
            return 1
        print("[PASS] RAG ingestion/retrieval")
    except Exception as e:
        print(f"[FAIL] RAG ingestion/retrieval: Error: {e}")
        return 1

    # Step D: Long-Term Semantic Memory Create & Search
    mem_scope = f"smoke_scope_{uuid.uuid4().hex[:6]}"
    mem_phrase = f"deployment-agent-pref-{uuid.uuid4().hex[:6]}"
    try:
        mem_payload = {
            "content": f"Operator preference for {mem_phrase}: automated health verification.",
            "scope_id": mem_scope,
            "memory_type": "user_preference",
            "importance": 0.85,
        }
        resp = client.post("/api/v1/memory", json=mem_payload)
        if resp.status_code != 201:
            print(f"[FAIL] Memory create: Expected 201, got {resp.status_code}")
            return 1

        # Search memory in the distinct scope
        search_mem_payload = {
            "query": mem_phrase,
            "scope_id": mem_scope,
            "top_k": 3,
        }
        resp = client.post("/api/v1/memory/search", json=search_mem_payload)
        if resp.status_code != 200:
            print(f"[FAIL] Memory search: Expected 200, got {resp.status_code}")
            return 1
        mem_results = resp.json().get("results", [])
        if not mem_results:
            print("[FAIL] Memory search: Created memory record was not returned in search.")
            return 1
        print("[PASS] Memory create/search")
    except Exception as e:
        print(f"[FAIL] Memory create/search: Error: {e}")
        return 1

    # Step E: Deterministic Orchestration / Checkpoint Execution
    thread_id = f"smoke-thread-{uuid.uuid4().hex[:8]}"
    try:
        run_payload: Dict[str, Any] = {
            "task": "Perform a deterministic calculation: 2 + 2",
            "thread_id": thread_id,
            "scope_id": mem_scope,
        }
        resp = client.post("/api/v1/orchestration/run", json=run_payload)
        if resp.status_code != 200:
            err_snippet = resp.text[:80]
            print(f"[FAIL] Orchestration: Expected 200, got {resp.status_code}: {err_snippet}")
            return 1

        run_data = resp.json()
        if run_data.get("status") not in {"completed", "interrupted"}:
            print(f"[FAIL] Orchestration/checkpoint: Unexpected status '{run_data.get('status')}'")
            return 1

        if not run_data.get("thread_id"):
            print("[FAIL] Orchestration/checkpoint: Missing thread_id in response")
            return 1

        if not run_data.get("answer"):
            print("[FAIL] Orchestration/checkpoint: Empty answer in response")
            return 1

        print("[PASS] Orchestration/checkpoint")
    except Exception as e:
        print(f"[FAIL] Orchestration/checkpoint: Error: {e}")
        return 1

    print("\nDeployment verification PASSED")
    return 0


def run_verification(
    base_url: str,
    timeout: float = 10.0,
    client: Optional[httpx.Client] = None,
) -> int:
    """Execute end-to-end smoke verification against a running deployment.

    Args:
        base_url: The root URL of the running API (e.g., http://localhost:8000).
        timeout: HTTP request timeout in seconds.
        client: Optional pre-configured httpx.Client (e.g. TestClient for in-process testing).

    Returns:
        0 if all deployment verification steps pass, 1 otherwise.
    """
    clean_base_url = base_url.rstrip("/")
    print(f"Starting deployment verification against: {clean_base_url}")

    if client is not None:
        return _run_checks(client)

    with httpx.Client(base_url=clean_base_url, timeout=timeout) as new_client:
        return _run_checks(new_client)


def main() -> None:
    """CLI entrypoint for deployment verification."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a RUNNING Multi-Agent Orchestration Platform deployment.\n"
            "Verifies HTTP liveness, readiness, RAG, memory, and orchestration.\n"
            "NOTE: This tests a running deployment, not container restart persistence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10.0)",
    )

    args = parser.parse_args()
    exit_code = run_verification(base_url=args.base_url, timeout=args.timeout)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
