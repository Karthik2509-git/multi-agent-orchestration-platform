"""Code specialist agent for programming, syntax inspection, and code generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.agents.base import AgentResult, BaseSpecializedAgent
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider

if TYPE_CHECKING:
    from src.app.orchestration.state import OrchestrationState

logger = get_logger(__name__)

CODE_SYSTEM_PROMPT = """You are a Code Specialist Agent specializing in software architecture,
code generation, code review, and algorithm design.
Provide clean, robust, type-annotated, and well-structured code or technical analysis.
IMPORTANT: You analyze and generate code only. You do NOT execute code or interact with the
system shell."""


class CodeAgent(BaseSpecializedAgent):
    """Specialized agent for code analysis, architecture design, and code generation.

    Note: This agent generates and analyzes code statically. It NEVER executes code
    or executes shell commands.
    """

    name: str = "code"

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, task: str, state: OrchestrationState) -> AgentResult:
        """Analyze or generate code according to the requested task and prior findings."""
        logger.info("CodeAgent executing code task: '%s'", task[:80])

        prior_results = state.get("agent_results", {})
        prior_context = ""
        if prior_results:
            prior_context = "\nPrior Agent Context:\n" + "\n".join(
                f"- [{k}]: {v[:250]}" for k, v in prior_results.items()
            )

        user_content = (
            f"Programming Task: {task}\n{prior_context}\n\n"
            "Please provide technical analysis, implementation, or review."
        )

        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self.provider.generate(messages=messages)
            result_text = response.content or "Code agent completed task with no output."
            return AgentResult(
                agent=self.name,
                status="success",
                result=result_text,
                metadata={"code_execution_attempted": False},
            )
        except Exception as e:
            logger.error("CodeAgent encountered error: %s", str(e))
            return AgentResult(
                agent=self.name,
                status="error",
                result=f"Code agent encountered error: {str(e)}",
            )
