"""Unit tests for MemoryExtractor."""

import pytest

from src.app.llm.providers.mock import MockLLMProvider
from src.app.memory.extraction import MemoryExtractor
from src.app.memory.models import MemoryType
from src.app.models.schemas.llm import LLMResponse


def test_deterministic_extraction():
    """Verify heuristic extraction detects preference directives and strategy summaries."""
    extractor = MemoryExtractor()
    memories = extractor.extract_deterministic(
        task="Please always use markdown tables when presenting tabular results.",
        final_answer="Here is the table summarizing results...",
        agent_results={"data": "table content computed"},
        scope_id="user_1",
        task_id="task_101",
    )

    assert len(memories) >= 1
    # Preference memory
    pref = next((m for m in memories if m.memory_type == MemoryType.USER_PREFERENCE), None)
    assert pref is not None
    assert "always" in pref.content.lower()
    assert pref.source_task_ids == ["task_101"]


@pytest.mark.asyncio
async def test_llm_based_extraction():
    """Verify LLM extraction parses valid JSON array into MemoryRecords."""
    mock_llm = MockLLMProvider(
        responses=[
            LLMResponse(
                content="""[
                    {
                        "content": "User requires CAGR formula to be explicitly cited.",
                        "memory_type": "user_preference",
                        "importance": 0.85
                    }
                ]""",
                model="mock-llm",
            )
        ]
    )

    extractor = MemoryExtractor(llm_provider=mock_llm)
    memories = await extractor.extract(
        task="Calculate Apple CAGR",
        final_answer="The CAGR is 2.37%",
        use_llm=True,
        scope_id="user_1",
    )

    assert len(memories) == 1
    assert memories[0].content == "User requires CAGR formula to be explicitly cited."
    assert memories[0].memory_type == MemoryType.USER_PREFERENCE
    assert memories[0].importance == 0.85
