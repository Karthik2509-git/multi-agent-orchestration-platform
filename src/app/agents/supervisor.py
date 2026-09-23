"""Supervisor agent responsible for routing tasks to specialized agents."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider

if TYPE_CHECKING:
    from src.app.orchestration.state import OrchestrationState

logger = get_logger(__name__)

ALLOWED_ROUTES = {"research", "data", "code", "final"}
RouteType = Literal["research", "data", "code", "final"]


class RouteDecision(BaseModel):
    """Validated routing decision produced by the supervisor."""

    next_agent: RouteType = Field(description="The target specialized agent or 'final' to finish")
    reasoning: Optional[str] = Field(
        default=None, description="Internal reasoning behind the routing choice"
    )


SUPERVISOR_SYSTEM_PROMPT = """You are an AI Orchestration Supervisor managing specialized
worker agents:
- 'research': handles gathering, researching, and analyzing information or documents.
- 'data': handles numerical calculation, statistical processing, and quantitative analysis.
- 'code': handles programming, code implementation, syntax analysis, and software engineering.
- 'final': select when all necessary work is complete, or when the task can be answered directly.

Your sole responsibility is to evaluate the user task and prior agent results, then select
the next agent.
Respond strictly in JSON format:
{
  "next_agent": "research" | "data" | "code" | "final",
  "reasoning": "Brief explanation of why this agent was selected"
}"""


class SupervisorAgent:
    """Supervisor node deciding task delegation across specialized agents."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _parse_routing_decision(self, response_text: Optional[str]) -> RouteDecision:
        """Strictly and defensively parse the model output into a validated RouteDecision."""
        if not response_text:
            logger.warning("Supervisor received empty response; falling back to 'final'")
            return RouteDecision(next_agent="final", reasoning="Empty model response")

        text = response_text.strip()
        # Clean markdown code block fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                raw_route = str(data.get("next_agent", "")).lower().strip()
                if raw_route in ALLOWED_ROUTES:
                    return RouteDecision(
                        next_agent=raw_route,  # type: ignore[arg-type]
                        reasoning=data.get("reasoning"),
                    )
                logger.warning(
                    "Supervisor model produced unauthorized route '%s'. Falling back to 'final'",
                    raw_route,
                )
                return RouteDecision(
                    next_agent="final",
                    reasoning=f"Unauthorized route '{raw_route}' defaulted to final",
                )
        except json.JSONDecodeError:
            # Fallback regex search for explicit keywords in case of partial JSON
            match = re.search(r'"next_agent"\s*:\s*"(\w+)"', text, re.IGNORECASE)
            if match:
                extracted = match.group(1).lower().strip()
                if extracted in ALLOWED_ROUTES:
                    return RouteDecision(
                        next_agent=extracted,  # type: ignore[arg-type]
                        reasoning="Extracted via regex fallback",
                    )

        logger.warning(
            "Supervisor failed to parse valid routing JSON from response: '%s'. "
            "Falling back to 'final'",
            text[:100],
        )
        return RouteDecision(
            next_agent="final", reasoning="Unparseable model output defaulted to final"
        )

    async def decide_route(self, state: OrchestrationState) -> RouteDecision:
        """Determine the next specialized agent to invoke based on current graph state."""
        task = state.get("task", "")
        agent_results = state.get("agent_results", {})
        step_count = state.get("step_count", 0)

        # Build context summary of previous work
        completed_work_summary = "\n".join(
            f"- [{agent_name}]: {result[:300]}..." for agent_name, result in agent_results.items()
        )
        if not completed_work_summary:
            completed_work_summary = "No previous agent results yet."

        user_prompt = (
            f"Original Task: {task}\n\n"
            f"Current Step: {step_count}\n\n"
            f"Completed Agent Work:\n{completed_work_summary}\n\n"
            "Which agent should be invoked next? Respond strictly with the JSON format."
        )

        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            "Supervisor evaluating route at step %d for task '%s'",
            step_count,
            task[:60],
        )
        llm_response = await self.provider.generate(messages=messages)
        decision = self._parse_routing_decision(llm_response.content)

        logger.info(
            "Supervisor routed task to '%s' (reasoning: %s)",
            decision.next_agent,
            decision.reasoning or "none",
        )
        return decision
