"""Multi-agent orchestration graph implementation using LangGraph with Memory and HITL."""

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.app.agents.code_agent import CodeAgent
from src.app.agents.data_agent import DataAgent
from src.app.agents.final_agent import FinalAgent
from src.app.agents.research_agent import ResearchAgent
from src.app.agents.supervisor import SupervisorAgent
from src.app.core.logging import get_logger
from src.app.hitl.policies import EscalationPolicy
from src.app.llm.base import LLMProvider
from src.app.memory.service import MemoryService
from src.app.observability import (
    record_agent_duration,
    record_agent_execution,
    record_agent_failure,
    record_hitl_escalation,
    record_span_error,
    set_span_attributes,
    trace_span,
)
from src.app.orchestration.state import OrchestrationState
from src.app.tools.calculator import CalculatorTool
from src.app.tools.http_tool import SafeHTTPGetTool

logger = get_logger(__name__)

# Maximum number of supervisor routing steps before terminating to final
MAX_ORCHESTRATION_STEPS = 8


def build_orchestration_graph(
    provider: LLMProvider,
    http_tool: Optional[SafeHTTPGetTool] = None,
    calculator: Optional[CalculatorTool] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    memory_service: Optional[MemoryService] = None,
    hitl_policy: Optional[EscalationPolicy] = None,
    tool_registry: Optional[Any] = None,
):
    """Build and compile the multi-agent LangGraph workflow with Memory, HITL, and Tools."""
    from src.app.tools.registry import ToolRegistry

    registry = tool_registry or ToolRegistry()
    if http_tool and http_tool.name not in registry:
        registry.register(http_tool)
    if calculator and calculator.name not in registry:
        registry.register(calculator)

    supervisor = SupervisorAgent(provider=provider)
    research_agent = ResearchAgent(provider=provider, http_tool=http_tool, registry=registry)
    data_agent = DataAgent(provider=provider, calculator=calculator, registry=registry)
    code_agent = CodeAgent(provider=provider, registry=registry)
    final_agent = FinalAgent(provider=provider)
    policy = hitl_policy or EscalationPolicy()

    async def memory_retrieval_node(state: OrchestrationState) -> Dict[str, Any]:
        """Retrieve relevant long-term memories before supervisor planning."""
        task = state.get("task", "")
        metadata = dict(state.get("metadata", {}))
        scope_id = metadata.get("scope_id", "default")
        memories_used: List[Dict[str, Any]] = []

        if memory_service and task:
            async with trace_span(
                "memory.retrieve",
                attributes={"scope_id": scope_id, "top_k": 3},
            ) as span:
                try:
                    results = await memory_service.search_memories(
                        query=task,
                        scope_id=scope_id,
                        top_k=3,
                    )
                    if results:
                        for r in results:
                            memories_used.append(
                                {
                                    "id": r.memory.id,
                                    "content": r.memory.content,
                                    "type": r.memory.memory_type.value,
                                    "importance": r.memory.importance,
                                    "score": r.score,
                                }
                            )
                        # Inject formatted memory context into messages
                        formatted_memories = memory_service.retriever.format_for_planning(results)
                        messages = list(state.get("messages", []))
                        messages.append(
                            {
                                "role": "system",
                                "content": f"[Long-Term Memory Context]\n{formatted_memories}",
                            }
                        )
                        set_span_attributes(
                            span,
                            {"hit_count": len(results), "status": "success"},
                        )
                        return {
                            "memories_used": memories_used,
                            "messages": messages,
                        }
                    set_span_attributes(span, {"hit_count": 0, "status": "success"})
                except Exception as e:
                    logger.warning("Memory retrieval before planning failed: %s", e)
                    set_span_attributes(span, {"status": "error", "error.type": type(e).__name__})

        return {"memories_used": memories_used}

    async def supervisor_node(state: OrchestrationState) -> Dict[str, Any]:
        step_count = state.get("step_count", 0)
        if step_count >= MAX_ORCHESTRATION_STEPS:
            logger.warning(
                "Reached orchestration step ceiling (%d). Forcing route to 'final'.",
                MAX_ORCHESTRATION_STEPS,
            )
            metadata = dict(state.get("metadata", {}))
            metadata["terminated_due_to_limit"] = True
            return {
                "next_agent": "final",
                "step_count": step_count + 1,
                "metadata": metadata,
            }

        async with trace_span(
            "supervisor.decide_route",
            attributes={"step_count": step_count},
        ) as span:
            decision = await supervisor.decide_route(state)
            set_span_attributes(
                span,
                {
                    "selected_route": decision.next_agent,
                    "allowed_route": decision.next_agent in ("research", "data", "code", "final"),
                    "status": "success",
                },
            )
            return {
                "next_agent": decision.next_agent,
                "step_count": step_count + 1,
            }

    def route_decision(state: OrchestrationState) -> str:
        next_agent = state.get("next_agent", "final")
        if next_agent in ("research", "data", "code"):
            return next_agent
        return "approval_gate"

    async def approval_gate_node(state: OrchestrationState) -> Dict[str, Any]:
        """Evaluate escalation policies and trigger interrupt if human approval is needed."""
        task = state.get("task", "")
        metadata = dict(state.get("metadata", {}))
        agent_results = state.get("agent_results", {})
        step_count = state.get("step_count", 0)

        # Evaluate escalation triggers
        should_escalate, reason, level = policy.evaluate(
            action_type="multi_agent_workflow",
            action_details={"task": task, "agent_results": agent_results},
            state_metadata=metadata,
            step_count=step_count,
        )

        if should_escalate:
            logger.info("HITL Approval Gate triggered: %s (level: %s)", reason, level.value)
            record_hitl_escalation(approval_level=level.value, reason=reason)
            findings_summary = list(agent_results.keys())
            interrupt_payload = {
                "interrupt_id": str(uuid4()),
                "action_type": "workflow_execution",
                "action_details": {"task": task, "agent_results": agent_results},
                "reason": reason,
                "context_summary": f"Task: {task}. Specialist findings: {findings_summary}",
                "level": level.value,
            }

            # Pause graph execution and wait for Command(resume=...)
            async with trace_span(
                "hitl.interrupt",
                attributes={
                    "interrupt_id": interrupt_payload["interrupt_id"],
                    "approval_level": level.value,
                    "reason_category": reason[:80],
                    "status": "interrupted",
                },
            ):
                human_decision = interrupt(interrupt_payload)

            if isinstance(human_decision, dict):
                decision_val = human_decision.get("decision", "approve")
                feedback = human_decision.get("feedback")
                override_output = human_decision.get("override_output")

                if decision_val == "reject":
                    logger.info("Human reviewer rejected action: %s", feedback)
                    metadata["human_decision"] = "rejected"
                    metadata["human_feedback"] = feedback
                    abort_msg = (
                        f"Task aborted by human reviewer: {feedback or 'No feedback provided.'}"
                    )
                    return {
                        "final_answer": abort_msg,
                        "status": "error",
                        "metadata": metadata,
                        "next_agent": "final",
                    }
                elif decision_val == "take_over":
                    logger.info("Human reviewer took over execution with direct answer.")
                    metadata["human_decision"] = "take_over"
                    metadata["human_feedback"] = feedback
                    return {
                        "final_answer": override_output or "Human reviewer completed task.",
                        "status": "completed",
                        "metadata": metadata,
                        "next_agent": "final",
                    }
                elif decision_val == "modify":
                    logger.info(
                        "Human reviewer modified action parameters: %s",
                        human_decision.get("modified_action"),
                    )
                    metadata["human_decision"] = "modified"
                    metadata["human_feedback"] = feedback
                    modified = human_decision.get("modified_action") or {}
                    metadata["modified_action"] = modified

                    # Apply modified task if provided in modified parameters
                    if isinstance(modified, dict):
                        if "task" in modified and modified["task"]:
                            task = str(modified["task"])
                        elif "modified_task" in modified and modified["modified_task"]:
                            task = str(modified["modified_task"])

                    return {
                        "task": task,
                        "metadata": metadata,
                    }

        return {"metadata": metadata}

    async def research_node(state: OrchestrationState) -> Dict[str, Any]:
        start_t = time.perf_counter()
        async with trace_span(
            "agent.execute",
            attributes={"agent_name": "research", "step_number": state.get("step_count", 0)},
        ) as span:
            try:
                result = await research_agent.run(state.get("task", ""), state)
                duration = time.perf_counter() - start_t
                record_agent_execution("research", result.status)
                record_agent_duration("research", duration)
                if result.status != "success":
                    record_agent_failure("research", "execution_error")
                set_span_attributes(
                    span,
                    {
                        "status": result.status,
                        "result_chars": len(result.result) if result.result else 0,
                    },
                )
                agent_results = dict(state.get("agent_results", {}))
                agent_results[result.agent] = result.result
                agents_used = list(state.get("agents_used", []))
                if result.agent not in agents_used:
                    agents_used.append(result.agent)
                return {
                    "agent_results": agent_results,
                    "agents_used": agents_used,
                }
            except Exception as e:
                duration = time.perf_counter() - start_t
                record_agent_execution("research", "error")
                record_agent_failure("research", e)
                record_agent_duration("research", duration)
                record_span_error(span, e)
                raise

    async def data_node(state: OrchestrationState) -> Dict[str, Any]:
        start_t = time.perf_counter()
        async with trace_span(
            "agent.execute",
            attributes={"agent_name": "data", "step_number": state.get("step_count", 0)},
        ) as span:
            try:
                result = await data_agent.run(state.get("task", ""), state)
                duration = time.perf_counter() - start_t
                record_agent_execution("data", result.status)
                record_agent_duration("data", duration)
                if result.status != "success":
                    record_agent_failure("data", "execution_error")
                set_span_attributes(
                    span,
                    {
                        "status": result.status,
                        "result_chars": len(result.result) if result.result else 0,
                    },
                )
                agent_results = dict(state.get("agent_results", {}))
                agent_results[result.agent] = result.result
                agents_used = list(state.get("agents_used", []))
                if result.agent not in agents_used:
                    agents_used.append(result.agent)
                return {
                    "agent_results": agent_results,
                    "agents_used": agents_used,
                }
            except Exception as e:
                duration = time.perf_counter() - start_t
                record_agent_execution("data", "error")
                record_agent_failure("data", e)
                record_agent_duration("data", duration)
                record_span_error(span, e)
                raise

    async def code_node(state: OrchestrationState) -> Dict[str, Any]:
        start_t = time.perf_counter()
        async with trace_span(
            "agent.execute",
            attributes={"agent_name": "code", "step_number": state.get("step_count", 0)},
        ) as span:
            try:
                result = await code_agent.run(state.get("task", ""), state)
                duration = time.perf_counter() - start_t
                record_agent_execution("code", result.status)
                record_agent_duration("code", duration)
                if result.status != "success":
                    record_agent_failure("code", "execution_error")
                set_span_attributes(
                    span,
                    {
                        "status": result.status,
                        "result_chars": len(result.result) if result.result else 0,
                    },
                )
                agent_results = dict(state.get("agent_results", {}))
                agent_results[result.agent] = result.result
                agents_used = list(state.get("agents_used", []))
                if result.agent not in agents_used:
                    agents_used.append(result.agent)
                return {
                    "agent_results": agent_results,
                    "agents_used": agents_used,
                }
            except Exception as e:
                duration = time.perf_counter() - start_t
                record_agent_execution("code", "error")
                record_agent_failure("code", e)
                record_agent_duration("code", duration)
                record_span_error(span, e)
                raise

    async def final_node(state: OrchestrationState) -> Dict[str, Any]:
        metadata = dict(state.get("metadata", {}))
        human_decision = metadata.get("human_decision")

        # If human takeover or rejection already produced final answer, preserve it
        if state.get("final_answer"):
            agents_used = list(state.get("agents_used", []))
            status_val = state.get("status", "completed")
            async with trace_span(
                "final.synthesis",
                attributes={
                    "status": status_val,
                    "answer_chars": len(state["final_answer"]),
                    "memory_extraction": False,
                },
            ):
                return {
                    "final_answer": state["final_answer"],
                    "status": status_val,
                    "agents_used": agents_used,
                    "metadata": metadata,
                }

        start_t = time.perf_counter()
        async with trace_span(
            "agent.execute",
            attributes={"agent_name": "final", "step_number": state.get("step_count", 0)},
        ) as agent_span:
            try:
                result = await final_agent.run(state.get("task", ""), state)
                duration = time.perf_counter() - start_t
                record_agent_execution("final", result.status)
                record_agent_duration("final", duration)
                if result.status != "success":
                    record_agent_failure("final", "execution_error")
                set_span_attributes(
                    agent_span,
                    {
                        "status": result.status,
                        "result_chars": len(result.result) if result.result else 0,
                    },
                )
                agents_used = list(state.get("agents_used", []))
                if result.agent not in agents_used:
                    agents_used.append(result.agent)

                final_answer = result.result
                status_val = "completed" if result.status == "success" else "error"

                will_extract = bool(
                    memory_service and status_val == "completed" and human_decision != "rejected"
                )

                async with trace_span(
                    "final.synthesis",
                    attributes={
                        "status": status_val,
                        "answer_chars": len(final_answer) if final_answer else 0,
                        "memory_extraction": will_extract,
                    },
                ):
                    # Extract and persist long-term memories if memory service is configured,
                    # status is completed, and the workflow was not rejected
                    if will_extract and memory_service:
                        try:
                            task = state.get("task", "")
                            scope_id = metadata.get("scope_id", "default")
                            await memory_service.extract_and_store_memories(
                                task=task,
                                final_answer=final_answer,
                                agent_results=state.get("agent_results"),
                                scope_id=scope_id,
                            )
                        except Exception as e:
                            logger.warning("Post-task memory extraction failed: %s", e)

                return {
                    "final_answer": final_answer,
                    "status": status_val,
                    "agents_used": agents_used,
                    "metadata": metadata,
                }
            except Exception as e:
                duration = time.perf_counter() - start_t
                record_agent_execution("final", "error")
                record_agent_failure("final", e)
                record_agent_duration("final", duration)
                record_span_error(agent_span, e)
                raise

    # Assemble StateGraph
    workflow = StateGraph(OrchestrationState)
    workflow.add_node("memory_retrieval", memory_retrieval_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("data", data_node)
    workflow.add_node("code", code_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "memory_retrieval")
    workflow.add_edge("memory_retrieval", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "research": "research",
            "data": "data",
            "code": "code",
            "approval_gate": "approval_gate",
        },
    )
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("data", "supervisor")
    workflow.add_edge("code", "supervisor")
    workflow.add_edge("approval_gate", "final")
    workflow.add_edge("final", END)

    return workflow.compile(checkpointer=checkpointer)
