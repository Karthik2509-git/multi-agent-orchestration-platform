"""Single AI agent implementing the core tool-calling execution loop."""

import json
import time
from typing import Any, Dict, List

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.models.schemas.agent import AgentRunResponse, ToolCallSummary
from src.app.tools.registry import ToolRegistry

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and precise AI assistant equipped with specialized tools. "
    "When a task requires computation or retrieving external information, "
    "invoke the appropriate tool. Once you have received the tool results, "
    "provide a clear, direct, and helpful final response."
)


class ToolCallingAgent:
    """Agent that coordinates tool selection, execution, and LLM synthesis."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_iterations: int = 5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.provider = provider
        self.registry = registry
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    async def run(self, task: str) -> AgentRunResponse:
        """Run the agent loop until the task is complete or max iterations is reached."""
        start_time = time.perf_counter()
        logger.info("Agent execution started for task: '%s'", task[:100])

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        tool_calls_summary: List[ToolCallSummary] = []
        tool_schemas = self.registry.get_schemas()
        iterations = 0
        last_model = getattr(self.provider, "model", "unknown")

        try:
            while iterations < self.max_iterations:
                iterations += 1
                logger.info(
                    "Agent iteration %d/%d invoking LLM",
                    iterations,
                    self.max_iterations,
                )

                llm_response = await self.provider.generate(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )
                if llm_response.model:
                    last_model = llm_response.model

                # If the LLM did not request any tools, it has reached the final answer
                if not llm_response.tool_calls:
                    duration = time.perf_counter() - start_time
                    logger.info(
                        "Agent produced final response in %.3fs (iterations=%d)",
                        duration,
                        iterations,
                    )
                    return AgentRunResponse(
                        task=task,
                        answer=llm_response.content or "",
                        tool_calls=tool_calls_summary,
                        model=last_model,
                        status="completed",
                        execution_time_seconds=round(duration, 4),
                    )

                # Process requested tool calls
                logger.info(
                    "LLM requested %d tool calls in iteration %d",
                    len(llm_response.tool_calls),
                    iterations,
                )

                # Record assistant message with tool calls in history
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in llm_response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Execute each tool and append results
                for tc in llm_response.tool_calls:
                    logger.info("Tool selected: '%s' with args %s", tc.name, tc.arguments)
                    tool_start = time.perf_counter()

                    tool_result = await self.registry.execute(tc.name, tc.arguments)
                    tool_duration_ms = round((time.perf_counter() - tool_start) * 1000, 2)

                    if tool_result.success:
                        logger.info(
                            "Tool '%s' succeeded in %.2fms",
                            tc.name,
                            tool_duration_ms,
                        )
                        content_payload = tool_result.data
                    else:
                        logger.warning(
                            "Tool '%s' failed in %.2fms: %s",
                            tc.name,
                            tool_duration_ms,
                            tool_result.error,
                        )
                        content_payload = {"error": tool_result.error}

                    # Append tool response message to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(content_payload),
                        }
                    )

                    # Record in audit summary
                    tool_calls_summary.append(
                        ToolCallSummary(
                            tool_name=tc.name,
                            arguments=tc.arguments,
                            result=tool_result.data,
                            success=tool_result.success,
                            error=tool_result.error,
                            duration_ms=tool_duration_ms,
                        )
                    )

            # Max iterations reached without a final response
            total_duration = time.perf_counter() - start_time
            logger.warning(
                "Agent reached maximum iterations limit (%d) for task: '%s'",
                self.max_iterations,
                task[:100],
            )
            return AgentRunResponse(
                task=task,
                answer=(
                    f"Agent reached maximum iteration limit ({self.max_iterations}) "
                    "without producing a final answer."
                ),
                tool_calls=tool_calls_summary,
                model=last_model,
                status="max_iterations_reached",
                execution_time_seconds=round(total_duration, 4),
            )

        except Exception as err:
            total_duration = time.perf_counter() - start_time
            logger.error("Agent execution failed with error: %s", str(err))
            return AgentRunResponse(
                task=task,
                answer=f"An error occurred during agent execution: {str(err)}",
                tool_calls=tool_calls_summary,
                model=last_model,
                status="error",
                execution_time_seconds=round(total_duration, 4),
            )
