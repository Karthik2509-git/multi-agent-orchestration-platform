"""Low-cardinality OpenTelemetry metrics layer for multi-agent orchestration.

Provides bounded, safe operational metrics across:
1. Orchestration
2. Agents
3. LLM calls
4. Tools
5. RAG
6. Memory
7. HITL

STRICT CARDINALITY RULE:
High-cardinality values (e.g. thread_id, task_id, scope_id, trace_id, span_id,
memory_id, document_id, chunk_id, request_id, user_ids, prompt/task text,
raw errors, URLs) are NEVER permitted as metric dimensions.
"""

from typing import Any, Optional

from src.app.core.logging import get_logger
from src.app.observability.cost import CostEstimate, CostStatus, calculate_llm_cost
from src.app.observability.telemetry import get_meter, get_telemetry_manager

logger = get_logger(__name__)

# Controlled dimension domains
ALLOWED_ORCHESTRATION_STATUSES = {"completed", "interrupted", "error"}
ALLOWED_ORCHESTRATION_ERROR_CATEGORIES = {
    "guardrail_rejection",
    "timeout",
    "unhandled_exception",
    "provider_error",
    "tool_error",
    "other",
}

ALLOWED_AGENT_NAMES = {"research", "data", "code", "final"}
ALLOWED_AGENT_STATUSES = {"success", "error"}
ALLOWED_AGENT_ERROR_CATEGORIES = {
    "timeout",
    "validation_error",
    "execution_error",
    "provider_error",
    "other",
}

ALLOWED_LLM_STATUSES = {"success", "error"}
ALLOWED_LLM_TOKEN_TYPES = {"prompt", "completion"}
ALLOWED_COST_STATUSES = {"known", "unknown", "unavailable"}

ALLOWED_TOOL_TYPES = {"native", "mcp"}
ALLOWED_TOOL_STATUSES = {"success", "error"}
ALLOWED_TOOL_ERROR_CATEGORIES = {
    "timeout",
    "validation_error",
    "execution_error",
    "budget_exceeded",
    "tool_disabled",
    "other",
}

ALLOWED_RAG_STRATEGIES = {"hybrid", "dense", "sparse"}
ALLOWED_RAG_STATUSES = {"success", "error"}

ALLOWED_MEMORY_STATUSES = {"success", "error"}
ALLOWED_MEMORY_HAS_HITS = {"true", "false"}

ALLOWED_APPROVAL_LEVELS = {"notify", "approve_action", "approve_plan", "take_over"}
ALLOWED_HITL_REASONS = {
    "sensitive_action",
    "low_confidence",
    "retry_ceiling",
    "explicit_review",
    "other",
}
ALLOWED_HITL_DECISIONS = {"approve", "reject", "modify", "take_over"}


class _MetricsRegistry:
    """Manages OpenTelemetry instruments dynamically bound to the active Meter."""

    def __init__(self) -> None:
        self._meter = None
        self._init_instruments()

    def _init_instruments(self) -> None:
        current_meter = get_meter()
        self._meter = current_meter

        # 1. Orchestration Metrics
        self.orchestration_requests = current_meter.create_counter(
            name="orchestration_requests",
            description="Total number of orchestration requests",
            unit="1",
        )
        self.orchestration_failures = current_meter.create_counter(
            name="orchestration_failures",
            description="Total number of orchestration failures",
            unit="1",
        )
        self.orchestration_duration = current_meter.create_histogram(
            name="orchestration_duration",
            description="Duration of orchestration requests in seconds",
            unit="s",
        )

        # 2. Agent Metrics
        self.agent_executions = current_meter.create_counter(
            name="agent_executions",
            description="Total number of agent node executions",
            unit="1",
        )
        self.agent_failures = current_meter.create_counter(
            name="agent_failures",
            description="Total number of agent execution failures",
            unit="1",
        )
        self.agent_duration = current_meter.create_histogram(
            name="agent_duration",
            description="Duration of agent execution in seconds",
            unit="s",
        )

        # 3. LLM Metrics
        self.llm_calls = current_meter.create_counter(
            name="llm_calls",
            description="Total number of LLM provider calls",
            unit="1",
        )
        self.llm_tokens = current_meter.create_counter(
            name="llm_tokens",
            description="Total number of actual LLM tokens consumed",
            unit="1",
        )
        self.llm_latency = current_meter.create_histogram(
            name="llm_latency",
            description="Latency of LLM calls in seconds",
            unit="s",
        )
        self.llm_cost_estimated = current_meter.create_counter(
            name="llm_cost_estimated",
            description="Estimated cost of LLM calls in USD",
            unit="USD",
        )

        # 4. Tool Metrics
        self.tool_calls = current_meter.create_counter(
            name="tool_calls",
            description="Total number of tool calls",
            unit="1",
        )
        self.tool_failures = current_meter.create_counter(
            name="tool_failures",
            description="Total number of tool execution failures",
            unit="1",
        )
        self.tool_latency = current_meter.create_histogram(
            name="tool_latency",
            description="Latency of tool execution in seconds",
            unit="s",
        )

        # 5. RAG Metrics
        self.rag_retrievals = current_meter.create_counter(
            name="rag_retrievals",
            description="Total number of RAG retrievals",
            unit="1",
        )
        self.rag_retrieval_latency = current_meter.create_histogram(
            name="rag_retrieval_latency",
            description="Latency of RAG retrievals in seconds",
            unit="s",
        )
        self.rag_chunks_retrieved = current_meter.create_counter(
            name="rag_chunks_retrieved",
            description="Total number of chunks retrieved by RAG",
            unit="1",
        )

        # 6. Memory Metrics
        self.memory_searches = current_meter.create_counter(
            name="memory_searches",
            description="Total number of memory search operations",
            unit="1",
        )
        self.memory_hits = current_meter.create_counter(
            name="memory_hits",
            description="Total number of memory search hits or misses",
            unit="1",
        )

        # 7. HITL Metrics
        self.hitl_escalations = current_meter.create_counter(
            name="hitl_escalations",
            description="Total number of HITL escalations",
            unit="1",
        )
        self.hitl_decisions = current_meter.create_counter(
            name="hitl_decisions",
            description="Total number of HITL decisions",
            unit="1",
        )
        self.hitl_approval_latency = current_meter.create_histogram(
            name="hitl_approval_latency",
            description="Latency of HITL approval decisions in seconds",
            unit="s",
        )

    def get(self) -> "_MetricsRegistry":
        current_meter = get_meter()
        if self._meter is not current_meter:
            self._init_instruments()
        return self


_registry = _MetricsRegistry()


# Error and reason category normalization helpers
def categorize_orchestration_error(error: Any) -> str:
    """Map error into controlled orchestration error categories."""
    msg = str(error).lower()
    if isinstance(error, TimeoutError) or "timeout" in msg:
        return "timeout"
    if "guardrail" in msg:
        return "guardrail_rejection"
    if "provider" in msg or "llm" in msg or "openai" in msg or "gemini" in msg or "groq" in msg:
        return "provider_error"
    if "tool" in msg:
        return "tool_error"
    if isinstance(error, Exception):
        return "unhandled_exception"
    return "other"


def categorize_agent_error(error: Any) -> str:
    """Map error into controlled agent error categories."""
    msg = str(error).lower()
    if isinstance(error, TimeoutError) or "timeout" in msg:
        return "timeout"
    if "validation" in msg or isinstance(error, (ValueError, TypeError)):
        return "validation_error"
    if "provider" in msg or "llm" in msg:
        return "provider_error"
    if isinstance(error, RuntimeError):
        return "execution_error"
    return "other"


def categorize_tool_error(error: Any) -> str:
    """Map error into controlled tool error categories."""
    msg = str(error).lower()
    if "budget" in msg:
        return "budget_exceeded"
    if "disabled" in msg:
        return "tool_disabled"
    if isinstance(error, TimeoutError) or "timeout" in msg:
        return "timeout"
    if "validation" in msg or "argument" in msg or isinstance(error, (ValueError, TypeError)):
        return "validation_error"
    if "execution" in msg or isinstance(error, RuntimeError):
        return "execution_error"
    return "other"


def categorize_hitl_reason(reason: Any) -> str:
    """Map raw HITL reason into controlled low-cardinality categories."""
    r = str(reason).lower()
    if "sensitive" in r or "financial" in r or "deletion" in r or "deployment" in r:
        return "sensitive_action"
    if "confidence" in r:
        return "low_confidence"
    if "retry" in r or "limit" in r or "ceiling" in r:
        return "retry_ceiling"
    if "review" in r or "authorization" in r or "flagged" in r or "user" in r:
        return "explicit_review"
    return "other"


# Safe Metric Recording API
def record_orchestration_request(status: str) -> None:
    """Record an orchestration request completion status."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        stat = status if status in ALLOWED_ORCHESTRATION_STATUSES else "other"
        _registry.get().orchestration_requests.add(1, {"status": stat})
    except Exception as e:
        logger.warning("Failed to record orchestration_requests metric: %s", e)


def record_orchestration_failure(error: Any) -> None:
    """Record an orchestration failure categorized by error type."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        category = categorize_orchestration_error(error)
        _registry.get().orchestration_failures.add(1, {"error_category": category})
    except Exception as e:
        logger.warning("Failed to record orchestration_failures metric: %s", e)


def record_orchestration_duration(duration_seconds: float, status: str) -> None:
    """Record the duration of an orchestration request."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        stat = status if status in ALLOWED_ORCHESTRATION_STATUSES else "completed"
        _registry.get().orchestration_duration.record(
            max(0.0, float(duration_seconds)), {"status": stat}
        )
    except Exception as e:
        logger.warning("Failed to record orchestration_duration metric: %s", e)


def record_agent_execution(agent_name: str, status: str) -> None:
    """Record an agent node execution."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        name = agent_name if agent_name in ALLOWED_AGENT_NAMES else "final"
        stat = status if status in ALLOWED_AGENT_STATUSES else "error"
        _registry.get().agent_executions.add(1, {"agent_name": name, "status": stat})
    except Exception as e:
        logger.warning("Failed to record agent_executions metric: %s", e)


def record_agent_failure(agent_name: str, error: Any) -> None:
    """Record an agent node failure categorized by error type."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        name = agent_name if agent_name in ALLOWED_AGENT_NAMES else "final"
        category = (
            error if str(error) in ALLOWED_AGENT_ERROR_CATEGORIES else categorize_agent_error(error)
        )
        _registry.get().agent_failures.add(1, {"agent_name": name, "error_category": category})
    except Exception as e:
        logger.warning("Failed to record agent_failures metric: %s", e)


def record_agent_duration(agent_name: str, duration_seconds: float) -> None:
    """Record agent execution latency in seconds."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        name = agent_name if agent_name in ALLOWED_AGENT_NAMES else "final"
        _registry.get().agent_duration.record(
            max(0.0, float(duration_seconds)), {"agent_name": name}
        )
    except Exception as e:
        logger.warning("Failed to record agent_duration metric: %s", e)


def record_llm_call(provider: str, model: str, status: str) -> None:
    """Record an LLM provider call count."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        prov = provider.strip().lower()
        mod = model.strip().lower()
        stat = status if status in ALLOWED_LLM_STATUSES else "error"
        _registry.get().llm_calls.add(1, {"provider": prov, "model": mod, "status": stat})
    except Exception as e:
        logger.warning("Failed to record llm_calls metric: %s", e)


def record_llm_latency(provider: str, model: str, latency_seconds: float) -> None:
    """Record an LLM call latency in seconds."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        prov = provider.strip().lower()
        mod = model.strip().lower()
        _registry.get().llm_latency.record(
            max(0.0, float(latency_seconds)), {"provider": prov, "model": mod}
        )
    except Exception as e:
        logger.warning("Failed to record llm_latency metric: %s", e)


def record_llm_tokens(provider: str, model: str, token_type: str, count: int) -> None:
    """Record actual LLM tokens consumed (only when reported by provider)."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        if count is None or count <= 0:
            return
        prov = provider.strip().lower()
        mod = model.strip().lower()
        tt = token_type if token_type in ALLOWED_LLM_TOKEN_TYPES else "prompt"
        _registry.get().llm_tokens.add(
            int(count), {"provider": prov, "model": mod, "token_type": tt}
        )
    except Exception as e:
        logger.warning("Failed to record llm_tokens metric: %s", e)


def record_llm_metrics(
    provider: str,
    model: str,
    latency_seconds: float,
    status: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    cost_estimate: Optional[CostEstimate] = None,
) -> None:
    """Record LLM call, token usage, latency, and estimated cost safely."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        reg = _registry.get()
        prov = provider.strip().lower()
        mod = model.strip().lower()
        stat = status if status in ALLOWED_LLM_STATUSES else "error"

        # 1. llm_calls
        reg.llm_calls.add(1, {"provider": prov, "model": mod, "status": stat})

        # 2. llm_latency
        reg.llm_latency.record(max(0.0, float(latency_seconds)), {"provider": prov, "model": mod})

        # 3. llm_tokens - ONLY record actual provider-reported counts. Never fabricate.
        if prompt_tokens is not None and prompt_tokens > 0:
            reg.llm_tokens.add(
                int(prompt_tokens),
                {"provider": prov, "model": mod, "token_type": "prompt"},
            )
        if completion_tokens is not None and completion_tokens > 0:
            reg.llm_tokens.add(
                int(completion_tokens),
                {"provider": prov, "model": mod, "token_type": "completion"},
            )

        # 4. llm_cost_estimated
        estimate = cost_estimate or calculate_llm_cost(
            provider=prov,
            model=mod,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if estimate.cost_status == CostStatus.KNOWN and estimate.estimated_cost_usd is not None:
            reg.llm_cost_estimated.add(
                float(estimate.estimated_cost_usd),
                {"provider": prov, "model": mod, "cost_status": "known"},
            )
        elif estimate.cost_status == CostStatus.UNKNOWN:
            # Record occurrence with 0.0 value and cost_status=unknown
            reg.llm_cost_estimated.add(
                0.0,
                {"provider": prov, "model": mod, "cost_status": "unknown"},
            )
        elif estimate.cost_status == CostStatus.UNAVAILABLE:
            reg.llm_cost_estimated.add(
                0.0,
                {"provider": prov, "model": mod, "cost_status": "unavailable"},
            )
    except Exception as e:
        logger.warning("Failed to record LLM metrics: %s", e)


def record_tool_call(tool_name: str, tool_type: str, status: str) -> None:
    """Record a tool execution call."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        tt = tool_type if tool_type in ALLOWED_TOOL_TYPES else "native"
        st = status if status in ALLOWED_TOOL_STATUSES else "error"
        _registry.get().tool_calls.add(1, {"tool_name": tool_name, "tool_type": tt, "status": st})
    except Exception as e:
        logger.warning("Failed to record tool_calls metric: %s", e)


def record_tool_failure(tool_name: str, error: Any) -> None:
    """Record a tool failure categorized by error type."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        cat = error if str(error) in ALLOWED_TOOL_ERROR_CATEGORIES else categorize_tool_error(error)
        _registry.get().tool_failures.add(1, {"tool_name": tool_name, "error_category": cat})
    except Exception as e:
        logger.warning("Failed to record tool_failures metric: %s", e)


def record_tool_latency(tool_name: str, tool_type: str, duration_seconds: float) -> None:
    """Record tool execution latency in seconds."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        tt = tool_type if tool_type in ALLOWED_TOOL_TYPES else "native"
        _registry.get().tool_latency.record(
            max(0.0, float(duration_seconds)),
            {"tool_name": tool_name, "tool_type": tt},
        )
    except Exception as e:
        logger.warning("Failed to record tool_latency metric: %s", e)


def record_rag_retrieval(strategy: str, status: str) -> None:
    """Record a RAG retrieval operation."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        strat = strategy if strategy in ALLOWED_RAG_STRATEGIES else "hybrid"
        st = status if status in ALLOWED_RAG_STATUSES else "error"
        _registry.get().rag_retrievals.add(1, {"strategy": strat, "status": st})
    except Exception as e:
        logger.warning("Failed to record rag_retrievals metric: %s", e)


def record_rag_latency(strategy: str, duration_seconds: float) -> None:
    """Record RAG retrieval latency in seconds."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        strat = strategy if strategy in ALLOWED_RAG_STRATEGIES else "hybrid"
        _registry.get().rag_retrieval_latency.record(
            max(0.0, float(duration_seconds)),
            {"strategy": strat},
        )
    except Exception as e:
        logger.warning("Failed to record rag_retrieval_latency metric: %s", e)


def record_rag_chunks_retrieved(strategy: str, count: int) -> None:
    """Record the count of chunks retrieved by RAG."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        strat = strategy if strategy in ALLOWED_RAG_STRATEGIES else "hybrid"
        _registry.get().rag_chunks_retrieved.add(max(0, int(count)), {"strategy": strat})
    except Exception as e:
        logger.warning("Failed to record rag_chunks_retrieved metric: %s", e)


def record_memory_search(status: str) -> None:
    """Record a memory search operation."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        st = status if status in ALLOWED_MEMORY_STATUSES else "error"
        _registry.get().memory_searches.add(1, {"status": st})
    except Exception as e:
        logger.warning("Failed to record memory_searches metric: %s", e)


def record_memory_hits(has_hits: bool) -> None:
    """Record whether a memory search resulted in hits."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        hit_str = "true" if has_hits else "false"
        _registry.get().memory_hits.add(1, {"has_hits": hit_str})
    except Exception as e:
        logger.warning("Failed to record memory_hits metric: %s", e)


def record_hitl_escalation(approval_level: str, reason: Any) -> None:
    """Record an escalation to human-in-the-loop."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        level = approval_level if approval_level in ALLOWED_APPROVAL_LEVELS else "approve_action"
        cat = reason if str(reason) in ALLOWED_HITL_REASONS else categorize_hitl_reason(reason)
        _registry.get().hitl_escalations.add(
            1,
            {"approval_level": level, "reason_category": cat},
        )
    except Exception as e:
        logger.warning("Failed to record hitl_escalations metric: %s", e)


def record_hitl_decision(decision: str) -> None:
    """Record a human-in-the-loop decision."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        dec = decision if decision in ALLOWED_HITL_DECISIONS else "approve"
        _registry.get().hitl_decisions.add(1, {"decision": dec})
    except Exception as e:
        logger.warning("Failed to record hitl_decisions metric: %s", e)


def record_hitl_approval_latency(decision: str, duration_seconds: float) -> None:
    """Record latency of a human-in-the-loop decision."""
    try:
        if not get_telemetry_manager().is_enabled:
            return
        dec = decision if decision in ALLOWED_HITL_DECISIONS else "approve"
        _registry.get().hitl_approval_latency.record(
            max(0.0, float(duration_seconds)),
            {"decision": dec},
        )
    except Exception as e:
        logger.warning("Failed to record hitl_approval_latency metric: %s", e)
