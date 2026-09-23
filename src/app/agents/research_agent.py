"""Research agent analyzing information and optionally fetching permitted URLs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from src.app.agents.base import AgentResult, BaseSpecializedAgent
from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.tools.http_tool import SafeHTTPGetTool

if TYPE_CHECKING:
    from src.app.orchestration.state import OrchestrationState

logger = get_logger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are a Research Agent specializing in analyzing information,
examining source material, and synthesizing research findings.
Provide a thorough, objective, and structured summary of your research findings
relevant to the user task.
If source data or documentation was retrieved, extract key facts, implications, and findings."""


class ResearchAgent(BaseSpecializedAgent):
    """Specialized agent for information analysis and permitted URL retrieval."""

    name: str = "research"

    def __init__(
        self,
        provider: LLMProvider,
        http_tool: Optional[SafeHTTPGetTool] = None,
    ):
        self.provider = provider
        self.http_tool = http_tool

    async def run(self, task: str, state: OrchestrationState) -> AgentResult:
        """Execute research task, optionally querying permitted URLs if mentioned."""
        logger.info("ResearchAgent starting research on task: '%s'", task[:80])
        retrieved_content: Optional[str] = None
        retrieved_url: Optional[str] = None

        # Check if the task explicitly provides an HTTP or HTTPS URL to inspect
        url_match = re.search(r"https?://[^\s\"'>]+", task)
        if url_match and self.http_tool:
            url_to_fetch = url_match.group(0)
            logger.info("ResearchAgent detected URL to inspect: %s", url_to_fetch)
            tool_res = await self.http_tool.execute(url=url_to_fetch)
            if tool_res.success and isinstance(tool_res.data, dict):
                retrieved_url = url_to_fetch
                retrieved_content = tool_res.data.get("content", "")
            else:
                logger.warning(
                    "ResearchAgent failed to fetch URL %s: %s",
                    url_to_fetch,
                    tool_res.error,
                )

        # Formulate synthesis prompt for the LLM
        prior_findings = state.get("agent_results", {})
        context_parts = [f"Task: {task}"]
        if prior_findings:
            context_parts.append(
                "Prior Findings from Other Agents: "
                + " | ".join(f"{k}: {v[:200]}" for k, v in prior_findings.items())
            )
        if retrieved_content:
            context_parts.append(
                f"Retrieved Document Content (from {retrieved_url}):\n{retrieved_content[:2000]}"
            )

        user_content = "\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self.provider.generate(messages=messages)
            result_text = (
                response.content or "Research agent completed analysis with no text output."
            )
            return AgentResult(
                agent=self.name,
                status="success",
                result=result_text,
                metadata={"url_inspected": retrieved_url} if retrieved_url else {},
            )
        except Exception as e:
            logger.error("ResearchAgent encountered error: %s", str(e))
            return AgentResult(
                agent=self.name,
                status="error",
                result=f"Research agent encountered error: {str(e)}",
            )
