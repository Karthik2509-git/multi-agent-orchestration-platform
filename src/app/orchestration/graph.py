"""Multi-agent orchestration graph implementation using LangGraph."""

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.app.agents.code_agent import CodeAgent
from src.app.agents.data_agent import DataAgent
from src.app.agents.final_agent import FinalAgent
from src.app.agents.research_agent import ResearchAgent
from src.app.agents.supervisor import SupervisorAgent
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
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
):
    """Build and compile the multi-agent LangGraph workflow."""
    supervisor = SupervisorAgent(provider=provider)
    research_agent = ResearchAgent(provider=provider, http_tool=http_tool)
    data_agent = DataAgent(provider=provider, calculator=calculator)
    code_agent = CodeAgent(provider=provider)
    final_agent = FinalAgent(provider=provider)

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

        decision = await supervisor.decide_route(state)
        return {
            "next_agent": decision.next_agent,
            "step_count": step_count + 1,
        }

    def route_decision(state: OrchestrationState) -> str:
        next_agent = state.get("next_agent", "final")
        if next_agent in ("research", "data", "code"):
            return next_agent
        return "final"

    async def research_node(state: OrchestrationState) -> Dict[str, Any]:
        result = await research_agent.run(state.get("task", ""), state)
        agent_results = dict(state.get("agent_results", {}))
        agent_results[result.agent] = result.result
        agents_used = list(state.get("agents_used", []))
        if result.agent not in agents_used:
            agents_used.append(result.agent)
        return {
            "agent_results": agent_results,
            "agents_used": agents_used,
        }

    async def data_node(state: OrchestrationState) -> Dict[str, Any]:
        result = await data_agent.run(state.get("task", ""), state)
        agent_results = dict(state.get("agent_results", {}))
        agent_results[result.agent] = result.result
        agents_used = list(state.get("agents_used", []))
        if result.agent not in agents_used:
            agents_used.append(result.agent)
        return {
            "agent_results": agent_results,
            "agents_used": agents_used,
        }

    async def code_node(state: OrchestrationState) -> Dict[str, Any]:
        result = await code_agent.run(state.get("task", ""), state)
        agent_results = dict(state.get("agent_results", {}))
        agent_results[result.agent] = result.result
        agents_used = list(state.get("agents_used", []))
        if result.agent not in agents_used:
            agents_used.append(result.agent)
        return {
            "agent_results": agent_results,
            "agents_used": agents_used,
        }

    async def final_node(state: OrchestrationState) -> Dict[str, Any]:
        result = await final_agent.run(state.get("task", ""), state)
        agents_used = list(state.get("agents_used", []))
        if result.agent not in agents_used:
            agents_used.append(result.agent)
        return {
            "final_answer": result.result,
            "status": "completed" if result.status == "success" else "error",
            "agents_used": agents_used,
        }

    # Assemble StateGraph
    workflow = StateGraph(OrchestrationState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("data", data_node)
    workflow.add_node("code", code_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "research": "research",
            "data": "data",
            "code": "code",
            "final": "final",
        },
    )
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("data", "supervisor")
    workflow.add_edge("code", "supervisor")
    workflow.add_edge("final", END)

    return workflow.compile()
