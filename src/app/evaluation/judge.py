"""Optional LLM-as-judge interface and implementations for semantic evaluation."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from src.app.core.logging import get_logger
from src.app.evaluation.models import JudgeEvaluationResult
from src.app.llm.base import LLMProvider

logger = get_logger(__name__)

DEFAULT_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluation judge evaluating answer quality,
accuracy, and groundedness.
Given a user query, retrieved context, and the assistant's answer, evaluate whether
the answer is accurate and properly supported by the context.

Respond ONLY with valid JSON in the following format:
{
  "score": <float between 0.0 and 1.0>,
  "passed": <boolean true or false>,
  "reasoning": "<clear concise explanation of judgment>"
}
"""


class BaseLLMJudge(ABC):
    """Abstract interface defining the contract for semantic evaluation judges."""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        criteria: Optional[str] = None,
    ) -> JudgeEvaluationResult:
        """Evaluate an answer against a query, context, and criteria."""
        pass


class MockLLMJudge(BaseLLMJudge):
    """Deterministic offline mock judge for unit testing and local benchmarks."""

    def __init__(
        self,
        default_score: float = 1.0,
        default_passed: bool = True,
        default_reasoning: str = "Deterministic mock judge evaluation passed.",
        custom_evaluator: Optional[
            Callable[[str, str, Optional[str], Optional[str]], JudgeEvaluationResult]
        ] = None,
    ):
        self.default_score = default_score
        self.default_passed = default_passed
        self.default_reasoning = default_reasoning
        self.custom_evaluator = custom_evaluator
        self.call_history: List[Dict[str, Any]] = []

    async def evaluate(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        criteria: Optional[str] = None,
    ) -> JudgeEvaluationResult:
        """Return deterministic mock evaluation result."""
        self.call_history.append(
            {"query": query, "answer": answer, "context": context, "criteria": criteria}
        )

        if self.custom_evaluator:
            return self.custom_evaluator(query, answer, context, criteria)

        return JudgeEvaluationResult(
            score=self.default_score,
            passed=self.default_passed,
            reasoning=self.default_reasoning,
            criteria=criteria,
            metadata={"mock": True, "evaluator": "MockLLMJudge"},
        )


class LLMJudge(BaseLLMJudge):
    """Semantic evaluation judge leveraging the application's LLMProvider abstraction."""

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: Optional[str] = None,
        pass_threshold: float = 0.7,
    ):
        self.provider = provider
        self.system_prompt = system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT
        self.pass_threshold = pass_threshold

    async def evaluate(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        criteria: Optional[str] = None,
    ) -> JudgeEvaluationResult:
        """Run LLM-based semantic evaluation safely without crashing."""
        user_prompt = f"Query: {query}\n\nAnswer: {answer}\n"
        if context:
            user_prompt += f"\nRetrieved Context:\n{context}\n"
        if criteria:
            user_prompt += f"\nEvaluation Criteria:\n{criteria}\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.provider.generate(messages=messages)
            raw_text = getattr(response, "content", None) or getattr(response, "text", "") or ""

            # Extract JSON block or raw JSON object
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                score = float(parsed.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                passed = bool(parsed.get("passed", score >= self.pass_threshold))
                reasoning = str(parsed.get("reasoning", "Evaluation completed."))
            else:
                score = 0.0
                passed = False
                reasoning = f"Failed to parse structured JSON from judge output: {raw_text[:100]}"

            return JudgeEvaluationResult(
                score=round(score, 4),
                passed=passed,
                reasoning=reasoning,
                criteria=criteria,
                metadata={"provider_response_id": getattr(response, "id", None)},
            )
        except Exception as e:
            logger.warning("LLMJudge evaluation failed: %s", e)
            return JudgeEvaluationResult(
                score=0.0,
                passed=False,
                reasoning=f"Judge execution error: {type(e).__name__}",
                criteria=criteria,
                metadata={"error": str(e)},
            )
