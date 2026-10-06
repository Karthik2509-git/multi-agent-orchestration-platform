"""Run-scoped execution context managing tool-call budgets and failure isolation.

Ensures:
1. A global run-scoped tool budget (default 10 calls) shared across all agents and tools.
2. The 11th attempted call is blocked before tool invocation.
3. Consecutive failure tracking per tool (default threshold: 3).
4. Run-scoped disablement of tools with 3 consecutive failures.
5. Successful execution resets consecutive failure streaks.
6. Async concurrency safety with lightweight run-local locking.
7. Seamless serialization compatibility with LangGraph checkpointing engines.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from src.app.core.logging import get_logger
from src.app.tools.base import ToolResult

logger = get_logger(__name__)


class ToolExecutionContext(dict):
    """Run-scoped context managing global tool budgets and failure isolation.

    Inherits from dict to provide zero-copy, native msgpack/JSON compatibility with
    LangGraph checkpointing, while exposing typed attributes, synchronization, and
    lifecycle methods.
    """

    def __init__(
        self,
        max_tool_calls: int = 10,
        consecutive_failure_threshold: int = 3,
        **kwargs: Any,
    ) -> None:
        if max_tool_calls <= 0:
            raise ValueError(f"max_tool_calls must be positive, got {max_tool_calls}")
        if consecutive_failure_threshold <= 0:
            raise ValueError(
                "consecutive_failure_threshold must be positive, "
                f"got {consecutive_failure_threshold}"
            )

        super().__init__(
            max_tool_calls=max_tool_calls,
            consecutive_failure_threshold=consecutive_failure_threshold,
            tool_calls_attempted=int(kwargs.get("tool_calls_attempted", 0)),
            tool_calls_succeeded=int(kwargs.get("tool_calls_succeeded", 0)),
            tool_calls_failed=int(kwargs.get("tool_calls_failed", 0)),
            failures_by_tool=dict(kwargs.get("failures_by_tool", {})),
            consecutive_failures_by_tool=dict(kwargs.get("consecutive_failures_by_tool", {})),
            disabled_tools=list(kwargs.get("disabled_tools", [])),
            tool_calls_by_name=dict(kwargs.get("tool_calls_by_name", {})),
        )
        self._lock: Optional[asyncio.Lock] = None

    @classmethod
    def ensure(cls, context_or_dict: Optional[Any]) -> ToolExecutionContext:
        """Coerce an existing ToolExecutionContext, dict, or None into a ToolExecutionContext."""
        if isinstance(context_or_dict, ToolExecutionContext):
            return context_or_dict
        if isinstance(context_or_dict, dict):
            return cls(**context_or_dict)
        return cls()

    @property
    def max_tool_calls(self) -> int:
        return self["max_tool_calls"]

    @max_tool_calls.setter
    def max_tool_calls(self, value: int) -> None:
        self["max_tool_calls"] = value

    @property
    def consecutive_failure_threshold(self) -> int:
        return self["consecutive_failure_threshold"]

    @consecutive_failure_threshold.setter
    def consecutive_failure_threshold(self, value: int) -> None:
        self["consecutive_failure_threshold"] = value

    @property
    def tool_calls_attempted(self) -> int:
        return self["tool_calls_attempted"]

    @tool_calls_attempted.setter
    def tool_calls_attempted(self, value: int) -> None:
        self["tool_calls_attempted"] = value

    @property
    def tool_calls_succeeded(self) -> int:
        return self["tool_calls_succeeded"]

    @tool_calls_succeeded.setter
    def tool_calls_succeeded(self, value: int) -> None:
        self["tool_calls_succeeded"] = value

    @property
    def tool_calls_failed(self) -> int:
        return self["tool_calls_failed"]

    @tool_calls_failed.setter
    def tool_calls_failed(self, value: int) -> None:
        self["tool_calls_failed"] = value

    @property
    def failures_by_tool(self) -> Dict[str, int]:
        return self["failures_by_tool"]

    @property
    def consecutive_failures_by_tool(self) -> Dict[str, int]:
        return self["consecutive_failures_by_tool"]

    @property
    def disabled_tools(self) -> Set[str]:
        return set(self["disabled_tools"])

    @property
    def tool_calls_by_name(self) -> Dict[str, int]:
        return self["tool_calls_by_name"]

    @property
    def lock(self) -> asyncio.Lock:
        """Provide an asyncio.Lock lazily initialized in the active event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check_and_reserve(self, tool_name: str) -> Optional[ToolResult]:
        """Check tool eligibility and reserve an attempt under the global budget.

        Order of enforcement:
        1. Tool disablement check: if disabled due to consecutive failures, block immediately.
        2. Attempt reservation: increment attempted count.
        3. Global budget check: if attempted exceeds max_tool_calls, block immediately.

        Returns:
            Optional[ToolResult]: A blocked ToolResult if rejected, or None if allowed to proceed.
        """
        async with self.lock:
            # 1. Check if tool is disabled for this run
            if tool_name in self.disabled_tools:
                logger.warning(
                    "Tool '%s' execution rejected: tool is disabled for this run",
                    tool_name,
                )
                return ToolResult(
                    success=False,
                    error=(
                        f"Tool '{tool_name}' is disabled for this run due to "
                        f"{self.consecutive_failure_threshold} consecutive failures."
                    ),
                    error_category="tool_disabled",
                )

            # 2. Reserve the attempt (counts all attempted calls)
            self.tool_calls_attempted += 1
            self.tool_calls_by_name[tool_name] = self.tool_calls_by_name.get(tool_name, 0) + 1

            # 3. Check global run budget
            if self.tool_calls_attempted > self.max_tool_calls:
                self.tool_calls_failed += 1
                logger.warning(
                    "Global tool budget exceeded on attempt %d (max: %d). Rejecting call to '%s'",
                    self.tool_calls_attempted,
                    self.max_tool_calls,
                    tool_name,
                )
                return ToolResult(
                    success=False,
                    error=(
                        f"Global tool-call budget exceeded (max: {self.max_tool_calls} allowed). "
                        "Execution blocked."
                    ),
                    error_category="budget_exceeded",
                )

            return None

    async def record_execution_result(self, tool_name: str, success: bool) -> None:
        """Update consecutive failure streaks, success totals, and auto-disablement state."""
        async with self.lock:
            if success:
                self.tool_calls_succeeded += 1
                # Successful execution resets the consecutive failure streak
                self.consecutive_failures_by_tool[tool_name] = 0
                logger.debug(
                    "Tool '%s' succeeded. Consecutive failure streak reset to 0.",
                    tool_name,
                )
            else:
                self.tool_calls_failed += 1
                self.failures_by_tool[tool_name] = self.failures_by_tool.get(tool_name, 0) + 1
                current_streak = self.consecutive_failures_by_tool.get(tool_name, 0) + 1
                self.consecutive_failures_by_tool[tool_name] = current_streak

                logger.warning(
                    "Tool '%s' failed (consecutive failures: %d/%d).",
                    tool_name,
                    current_streak,
                    self.consecutive_failure_threshold,
                )

                if current_streak >= self.consecutive_failure_threshold:
                    disabled_list: List[str] = self["disabled_tools"]
                    if tool_name not in disabled_list:
                        disabled_list.append(tool_name)
                    logger.error(
                        "Tool '%s' disabled for this run after %d consecutive failures.",
                        tool_name,
                        current_streak,
                    )

    def is_disabled(self, tool_name: str) -> bool:
        """Return True if the tool is disabled for this run."""
        return tool_name in self.disabled_tools

    def get_consecutive_failures(self, tool_name: str) -> int:
        """Return the current consecutive failure streak for the tool."""
        return self.consecutive_failures_by_tool.get(tool_name, 0)
