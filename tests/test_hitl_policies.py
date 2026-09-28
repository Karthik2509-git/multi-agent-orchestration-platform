"""Unit tests for HITL EscalationPolicy."""

from src.app.hitl.models import ApprovalLevel
from src.app.hitl.policies import EscalationPolicy


def test_deterministic_flag_trigger():
    """Verify requires_human_approval flag triggers escalation deterministically."""
    policy = EscalationPolicy()

    # In action details
    should, reason, level = policy.evaluate(
        action_type="generic_tool",
        action_details={"requires_human_approval": True},
    )
    assert should is True
    assert "explicitly flagged" in reason
    assert level == ApprovalLevel.APPROVE_ACTION

    # In state metadata
    should_meta, reason_meta, _ = policy.evaluate(
        action_type="generic_tool",
        state_metadata={"requires_human_approval": True},
    )
    assert should_meta is True


def test_sensitive_action_trigger():
    """Verify designated sensitive action types trigger escalation."""
    policy = EscalationPolicy(sensitive_actions=["data_deletion", "financial_transfer"])

    should, reason, level = policy.evaluate(action_type="financial_transfer")
    assert should is True
    assert "sensitive operation" in reason
    assert level == ApprovalLevel.APPROVE_ACTION


def test_max_retries_error_trigger():
    """Verify exceeding retry limit with error triggers takeover."""
    policy = EscalationPolicy(max_retries=2)

    should, reason, level = policy.evaluate(
        action_type="code_agent",
        action_details={"status": "error"},
        step_count=2,
    )
    assert should is True
    assert "retry limit" in reason
    assert level == ApprovalLevel.TAKE_OVER


def test_calibrated_confidence_trigger():
    """Verify confidence below threshold triggers review when explicitly provided."""
    policy = EscalationPolicy(confidence_threshold=0.70)

    # Low confidence -> triggers
    should_low, reason_low, level_low = policy.evaluate(action_type="plan", confidence=0.55)
    assert should_low is True
    assert "confidence" in reason_low

    # High confidence -> does not trigger
    should_high, _, _ = policy.evaluate(action_type="plan", confidence=0.90)
    assert should_high is False


def test_normal_action_no_escalation():
    """Verify safe action without triggers proceeds normally."""
    policy = EscalationPolicy()
    should, _, level = policy.evaluate(action_type="calculator", action_details={"query": "2 + 2"})
    assert should is False
    assert level == ApprovalLevel.NOTIFY
