"""Domain models and data structures for Human-in-the-Loop (HITL) workflows."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ApprovalLevel(str, Enum):
    """Level of approval required for an escalated action."""

    NOTIFY = "notify"
    APPROVE_ACTION = "approve_action"
    APPROVE_PLAN = "approve_plan"
    TAKE_OVER = "take_over"


class ApprovalDecision(str, Enum):
    """Human reviewer decision on an escalated action."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    TAKE_OVER = "take_over"


class HITLRequest(BaseModel):
    """Structured request emitted when LangGraph pauses on an interrupt."""

    interrupt_id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str = Field(..., description="Target LangGraph execution thread ID")
    action_type: str = Field(..., description="Classification of the paused action")
    action_details: Dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(..., description="Explanation of why human approval is required")
    context_summary: str = Field(default="", description="Summary of relevant task context")
    level: ApprovalLevel = Field(default=ApprovalLevel.APPROVE_ACTION)
    available_decisions: List[ApprovalDecision] = Field(
        default_factory=lambda: [
            ApprovalDecision.APPROVE,
            ApprovalDecision.REJECT,
            ApprovalDecision.MODIFY,
            ApprovalDecision.TAKE_OVER,
        ]
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HITLResponse(BaseModel):
    """Decision submitted by human reviewer to resume execution."""

    decision: ApprovalDecision = Field(..., description="Decision made by human reviewer")
    feedback: Optional[str] = Field(
        default=None, description="Optional explanation or instructions"
    )
    modified_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Modified action payload if decision is MODIFY",
    )
    override_output: Optional[str] = Field(
        default=None,
        description="Direct answer or synthetic output if decision is TAKE_OVER",
    )
