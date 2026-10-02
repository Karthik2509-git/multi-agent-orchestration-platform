"""Telemetry data sanitization and sensitive payload redaction."""

from typing import Any, Dict, Optional, Set

# Keys that indicate credentials or private secrets; NEVER recorded in telemetry attributes
SENSITIVE_KEYS: Set[str] = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "set_cookie",
    "proxy_authorization",
    "private_key",
    "access_token",
    "refresh_token",
}

# Payload keys that carry large or confidential prompt/response text
PAYLOAD_CONTENT_KEYS: Set[str] = {
    "prompt",
    "messages",
    "user_task",
    "task_text",
    "raw_task",
    "completion",
    "response_text",
    "generated_code",
    "code_content",
    "document_content",
    "chunk_content",
    "memory_content",
    "memory_text",
}


def clean_text_length(text: Optional[Any]) -> int:
    """Return character count of text safely without retaining content."""
    if text is None:
        return 0
    return len(str(text))


def is_sensitive_key(key: str) -> bool:
    """Check if key name matches any known credential or secret pattern."""
    normalized = key.lower().replace("-", "_").strip()
    return any(sens in normalized for sens in SENSITIVE_KEYS)


def sanitize_attributes(
    attributes: Dict[str, Any],
    record_payloads: bool = False,
) -> Dict[str, Any]:
    """Sanitize span attributes to prevent credential leakage and payload bloat.

    Args:
        attributes: Raw dictionary of candidate span attributes.
        record_payloads: If False (standard default), raw prompts, outputs, and
            payload contents are omitted, retaining only lengths and counts.

    Returns:
        Clean dictionary adhering to OpenTelemetry attribute type requirements.
    """
    sanitized: Dict[str, Any] = {}

    for key, val in attributes.items():
        if val is None:
            continue

        # Strictly drop any credential or sensitive header attributes
        if is_sensitive_key(key):
            continue

        norm_key = key.lower().replace("-", "_").strip()

        # Check for large/sensitive payload contents
        if any(payload_key in norm_key for payload_key in PAYLOAD_CONTENT_KEYS):
            # Record character length instead of raw content
            length_key = f"{key}_chars" if not key.endswith(("_chars", "_len", "_count")) else key
            sanitized[length_key] = clean_text_length(val)
            if record_payloads and isinstance(val, str):
                # Only when record_payloads is explicitly allowed (dev/test only)
                sanitized[f"{key}_preview"] = val[:200]
            continue

        # OpenTelemetry attributes support str, bool, int, float, and sequences thereof
        if isinstance(val, (str, bool, int, float)):
            # If string is excessively long (>1000 chars), truncate or convert to length
            if isinstance(val, str) and len(val) > 1000:
                sanitized[f"{key}_chars"] = len(val)
                if record_payloads:
                    sanitized[key] = val[:500]
            else:
                sanitized[key] = val
        elif isinstance(val, (list, tuple)):
            # Check if sequence contains only primitives
            if all(isinstance(item, (str, bool, int, float)) for item in val):
                sanitized[key] = list(val)
            else:
                sanitized[f"{key}_count"] = len(val)
        elif isinstance(val, dict):
            # For dictionaries, store count or clean summary rather than raw dict
            sanitized[f"{key}_count"] = len(val)
        else:
            sanitized[key] = str(val)[:200]

    return sanitized


def sanitize_error(error: Any) -> Dict[str, str]:
    """Extract safe error type and sanitized message for telemetry recording."""
    if isinstance(error, Exception):
        err_type = type(error).__name__
        err_msg = str(error)
    else:
        err_type = "Error"
        err_msg = str(error)

    # Truncate message and scrub sensitive tokens if accidentally included
    for sens in SENSITIVE_KEYS:
        if sens in err_msg.lower():
            err_msg = "[REDACTED_ERROR_DETAILS]"
            break

    return {
        "error.type": err_type,
        "error.message": err_msg[:300],
    }
