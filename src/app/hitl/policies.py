"""Escalation policies and trigger evaluation for Human-in-the-Loop workflows."""

from typing import Any, Dict, List, Optional, Tuple

from src.app.hitl.models import ApprovalLevel


class EscalationPolicy:
    """Evaluates whether an agent action requires human approval."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        max_retries: int = 2,
        sensitive_actions: Optional[List[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.sensitive_actions = sensitive_actions or [
            "financial_transaction",
            "data_deletion",
            "production_deployment",
            "external_api_dispatch",
        ]

    def evaluate(
        self,
        action_type: str,
        action_details: Optional[Dict[str, Any]] = None,
        state_metadata: Optional[Dict[str, Any]] = None,
        step_count: int = 0,
        confidence: Optional[float] = None,
    ) -> Tuple[bool, str, ApprovalLevel]:
        """Evaluate escalation rules and return (should_escalate, reason, level)."""
        details = action_details or {}
        metadata = state_metadata or {}

        # 1. Deterministic Trigger: explicit metadata flag on action or state
        if (
            details.get("requires_human_approval") is True
            or metadata.get("requires_human_approval") is True
        ):
            return (
                True,
                "Action explicitly flagged as requiring human authorization.",
                ApprovalLevel.APPROVE_ACTION,
            )

        if metadata.get("require_human_review") is True:
            return (
                True,
                "User requested explicit human review for this task.",
                ApprovalLevel.APPROVE_PLAN,
            )

        # 2. Sensitive Action Type Trigger
        if action_type in self.sensitive_actions or details.get("sensitive") is True:
            return (
                True,
                f"Action '{action_type}' is designated as a sensitive operation.",
                ApprovalLevel.APPROVE_ACTION,
            )

        # 3. Maximum Step / Retry Ceiling Trigger
        if step_count >= self.max_retries and details.get("status") == "error":
            return (
                True,
                f"Specialist exceeded maximum retry limit ({self.max_retries}).",
                ApprovalLevel.TAKE_OVER,
            )

        # 4. Optional Calibrated Confidence Trigger (only when explicit score is provided)
        if confidence is not None and confidence <= self.confidence_threshold:
            msg = (
                f"Agent confidence score ({confidence:.2f}) is below threshold "
                f"({self.confidence_threshold:.2f})."
            )
            return (
                True,
                msg,
                ApprovalLevel.APPROVE_PLAN,
            )

        return False, "", ApprovalLevel.NOTIFY
