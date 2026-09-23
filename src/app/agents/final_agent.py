"""Final aggregator agent synthesizing specialized agent findings into a final response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.agents.base import AgentResult, BaseSpecializedAgent
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider

if TYPE_CHECKING:
    from src.app.orchestration.state import OrchestrationState

logger = get_logger(__name__)

FINAL_SYSTEM_PROMPT = """You are the Final Response Synthesizer for an AI Multi-Agent
Orchestration Platform.
Your task is to take the user's initial objective and findings gathered by specialized
agents (research, data, code), and produce a comprehensive, coherent, and polished final answer.
- Answer the user's core question directly.
- Synthesize all relevant findings without repeating low-level internal coordination details.
- If no previous agent results were needed, answer the task thoroughly and directly."""


class FinalAgent(BaseSpecializedAgent):
    """Aggregator agent creating the final user-facing response."""

    name: str = "final"

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, task: str, state: OrchestrationState) -> AgentResult:
        """Synthesize accumulated agent results into a cohesive final answer."""
        logger.info("FinalAgent producing final synthesis for task: '%s'", task[:80])

        agent_results = state.get("agent_results", {})

        if agent_results:
            results_summary = "\n\n".join(
                f"### Findings from [{agent_name.upper()} AGENT]:\n{result}"
                for agent_name, result in agent_results.items()
            )
            user_content = (
                f"Original Task: {task}\n\n"
                f"Specialist Agent Findings:\n{results_summary}\n\n"
                "Please synthesize these findings into a unified, high-quality final answer."
            )
        else:
            user_content = (
                f"Task: {task}\n\nPlease provide a comprehensive and direct answer to this task."
            )

        messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self.provider.generate(messages=messages)
            answer = response.content or "No final answer produced."
            return AgentResult(
                agent=self.name,
                status="success",
                result=answer,
                metadata={"aggregated_agents": list(agent_results.keys())},
            )
        except Exception as e:
            logger.error("FinalAgent encountered error during synthesis: %s", str(e))
            return AgentResult(
                agent=self.name,
                status="error",
                result=f"Final synthesis error: {str(e)}",
            )
