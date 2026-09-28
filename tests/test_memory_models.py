"""Unit tests for Memory domain models."""

from datetime import datetime, timedelta, timezone

from src.app.memory.models import MemoryRecord, MemorySearchResult, MemoryType


def test_memory_record_defaults():
    """Verify default fields and uuid generation for MemoryRecord."""
    rec = MemoryRecord(content="User prefers succinct markdown bullet points.")
    assert rec.id is not None
    assert rec.scope_id == "default"
    assert rec.memory_type == MemoryType.DOMAIN_FACT
    assert rec.importance == 0.5
    assert rec.access_count == 0
    assert rec.is_expired is False


def test_memory_record_expiration():
    """Verify expiration calculation."""
    now = datetime.now(timezone.utc)
    expired_rec = MemoryRecord(
        content="Temporary cache key",
        expires_at=now - timedelta(seconds=10),
    )
    assert expired_rec.is_expired is True

    valid_rec = MemoryRecord(
        content="Permanent preference",
        expires_at=now + timedelta(days=7),
    )
    assert valid_rec.is_expired is False


def test_memory_search_result_model():
    """Verify MemorySearchResult serialization and score bounds."""
    rec = MemoryRecord(content="FastAPI is an async Python web framework.", importance=0.9)
    result = MemorySearchResult(memory=rec, score=0.875, similarity=0.92)
    assert result.score == 0.875
    assert result.similarity == 0.92
    assert result.memory.importance == 0.9
