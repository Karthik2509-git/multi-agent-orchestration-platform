"""Shared typed state for multi-agent orchestration graphs."""

from typing import Any, Dict, List, TypedDict


class OrchestrationState(TypedDict, total=False):
    """Shared state dictionary passed between nodes in the LangGraph workflow."""

    task: str
    messages: List[Dict[str, Any]]
    next_agent: str
    agent_results: Dict[str, str]
    agents_used: List[str]
    step_count: int
    final_answer: str
    status: str
    metadata: Dict[str, Any]
