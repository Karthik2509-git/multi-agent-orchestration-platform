"""Post-task memory extraction engine for distilling reusable insights."""

import json
import re
from typing import Dict, List, Optional
from uuid import uuid4

from src.app.core.logging import get_logger
from src.app.llm.base import LLMProvider
from src.app.memory.models import MemoryRecord, MemoryType

logger = get_logger(__name__)


class MemoryExtractor:
    """Extracts candidate memory records from completed agent workflows."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider

    def extract_deterministic(
        self,
        task: str,
        final_answer: str,
        agent_results: Optional[Dict[str, str]] = None,
        scope_id: str = "default",
        task_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Heuristic/deterministic extractor for offline testing and fast non-LLM pipelines."""
        memories: List[MemoryRecord] = []
        source_task_ids = [task_id] if task_id else []

        # 1. Check for user preference patterns (e.g. "prefer", "always", "format with")
        pref_match = re.search(
            r"(prefer|always|format as|use|never)\s+([^.\n]+)", task, re.IGNORECASE
        )
        if pref_match:
            pref_text = pref_match.group(0).strip()
            memories.append(
                MemoryRecord(
                    id=str(uuid4()),
                    scope_id=scope_id,
                    content=f"User directive: {pref_text}",
                    memory_type=MemoryType.USER_PREFERENCE,
                    importance=0.8,
                    source_task_ids=source_task_ids,
                    metadata={"extracted_by": "heuristic_rule", "rule": "preference_pattern"},
                )
            )

        # 2. Extract successful strategy/summary from agent results
        if agent_results:
            tools_used = list(agent_results.keys())
            summary_content = (
                f"Workflow on '{task[:60]}' successfully combined agents: {', '.join(tools_used)}."
            )
            memories.append(
                MemoryRecord(
                    id=str(uuid4()),
                    scope_id=scope_id,
                    content=summary_content,
                    memory_type=MemoryType.SUCCESSFUL_STRATEGY,
                    importance=0.6,
                    source_task_ids=source_task_ids,
                    metadata={"agents_used": tools_used, "extracted_by": "heuristic_rule"},
                )
            )

        return memories

    async def extract(
        self,
        task: str,
        final_answer: str,
        agent_results: Optional[Dict[str, str]] = None,
        scope_id: str = "default",
        task_id: Optional[str] = None,
        use_llm: bool = False,
    ) -> List[MemoryRecord]:
        """Extract memory items from task execution."""
        if not use_llm or not self.llm_provider:
            return self.extract_deterministic(
                task=task,
                final_answer=final_answer,
                agent_results=agent_results,
                scope_id=scope_id,
                task_id=task_id,
            )

        # LLM extraction path
        system_prompt = (
            "You are a memory extraction component for a multi-agent AI system. "
            "Analyze the completed task and output a JSON array of extracted long-term memories. "
            "Valid memory_type values: 'user_preference', 'task_lesson', "
            "'successful_strategy', 'domain_fact'. "
            "Each item must have: 'content' (string), 'memory_type' (string), "
            "'importance' (float between 0.0 and 1.0). "
            "Respond ONLY with valid JSON."
        )

        user_prompt = (
            f"Task: {task}\n\n"
            f"Agent Findings: {json.dumps(agent_results or {})}\n\n"
            f"Final Answer: {final_answer}\n\n"
            "Extract 1-3 useful, durable memory items."
        )

        try:
            response = await self.llm_provider.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            # Parse JSON
            raw_text = response.content.strip()
            # Clean markdown fences if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            items = json.loads(raw_text)
            if not isinstance(items, list):
                items = [items]

            records: List[MemoryRecord] = []
            source_task_ids = [task_id] if task_id else []

            for item in items:
                m_type = MemoryType(item.get("memory_type", MemoryType.DOMAIN_FACT.value))
                records.append(
                    MemoryRecord(
                        id=str(uuid4()),
                        scope_id=scope_id,
                        content=str(item.get("content", "")).strip(),
                        memory_type=m_type,
                        importance=float(item.get("importance", 0.5)),
                        source_task_ids=source_task_ids,
                        metadata={"extracted_by": "llm"},
                    )
                )

            return records
        except Exception as e:
            logger.warning(
                "LLM memory extraction failed (%s). Falling back to deterministic extractor.", e
            )
            return self.extract_deterministic(
                task=task,
                final_answer=final_answer,
                agent_results=agent_results,
                scope_id=scope_id,
                task_id=task_id,
            )
