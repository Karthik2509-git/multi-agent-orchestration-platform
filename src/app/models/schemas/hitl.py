"""Pydantic schemas for Human-in-the-Loop (HITL) API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.app.hitl.models import ApprovalDecision, HITLRequest


class PendingApprovalResponse(BaseModel):
    """Response schema when inspecting pending approval for an interrupted thread."""

    thread_id: str
    has_pending_approval: bool
    request: Optional[HITLRequest] = None


class ResumeTaskRequest(BaseModel):
    """Request schema for resuming an interrupted task."""

    decision: ApprovalDecision = Field(
        ..., description="Decision: 'approve', 'reject', 'modify', 'take_over'"
    )
    feedback: Optional[str] = Field(default=None, description="Optional human feedback or reason")
    modified_action: Optional[Dict[str, Any]] = Field(
        default=None, description="Modified parameters if modifying action"
    )
    override_output: Optional[str] = Field(
        default=None, description="Direct replacement output if taking over"
    )


class ResumeTaskResponse(BaseModel):
    """Response schema after resuming an interrupted thread."""

    thread_id: str
    status: str
    answer: str
    agents_used: List[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    pending_approval: Optional[HITLRequest] = None
    metadata: Optional[Dict[str, Any]] = None
