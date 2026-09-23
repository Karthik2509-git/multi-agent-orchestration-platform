"""Data agent handling calculations, quantitative evaluation, and analytical processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.app.agents.base import AgentResult, BaseSpecializedAgent
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.tools.calculator import CalculatorTool

if TYPE_CHECKING:
    from src.app.orchestration.state import OrchestrationState

logger = get_logger(__name__)

DATA_SYSTEM_PROMPT = """You are a Data Analysis Agent specializing in quantitative reasoning,
mathematical analysis, and data calculations.
When performing calculations, provide clear step-by-step reasoning, intermediate
numerical results, and summary conclusions."""


class DataAgent(BaseSpecializedAgent):
    """Specialized agent for mathematical computations and data analysis."""

    name: str = "data"

    def __init__(
        self,
        provider: LLMProvider,
        calculator: Optional[CalculatorTool] = None,
    ):
        self.provider = provider
        self.calculator = calculator or CalculatorTool()

    async def run(self, task: str, state: OrchestrationState) -> AgentResult:
        """Execute mathematical and analytical processing for the task."""
        logger.info("DataAgent executing analytical processing for: '%s'", task[:80])

        prior_results = state.get("agent_results", {})
        prior_context = ""
        if prior_results:
            prior_context = "\nPrior Agent Context:\n" + "\n".join(
                f"- {k}: {v[:250]}" for k, v in prior_results.items()
            )

        user_content = (
            f"Analytical Task: {task}\n{prior_context}\n\n"
            "Please analyze the data and calculate necessary values."
        )

        messages = [
            {"role": "system", "content": DATA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self.provider.generate(
                messages=messages,
                tools=[self.calculator.to_openai_schema()],
            )

            # If model requested a calculation tool call, execute it safely
            calc_summary = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    if tc.name == "calculator":
                        calc_res = await self.calculator.execute(**tc.arguments)
                        val = (
                            calc_res.data.get("result")
                            if isinstance(calc_res.data, dict)
                            else calc_res.data
                        )
                        outcome = val if calc_res.success else calc_res.error
                        calc_summary.append(
                            f"Evaluated {tc.arguments.get('expression')}: {outcome}"
                        )

            output_parts = []
            if calc_summary:
                output_parts.append("Calculations:\n" + "\n".join(calc_summary))
            if response.content:
                output_parts.append(response.content)

            final_text = "\n\n".join(output_parts) if output_parts else "Data analysis completed."
            return AgentResult(
                agent=self.name,
                status="success",
                result=final_text,
                metadata={"calculations_performed": len(calc_summary)},
            )
        except Exception as e:
            logger.error("DataAgent encountered error: %s", str(e))
            return AgentResult(
                agent=self.name,
                status="error",
                result=f"Data analysis error: {str(e)}",
            )
