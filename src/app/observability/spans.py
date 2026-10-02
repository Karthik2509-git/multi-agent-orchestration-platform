"""Span creation utilities, context propagation, and attribute helpers."""

from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Dict, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.app.observability.redaction import sanitize_attributes, sanitize_error
from src.app.observability.telemetry import get_telemetry_manager, get_tracer


def get_current_trace_id() -> Optional[str]:
    """Return active trace_id formatted as a 32-character hex string if active."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context() if current_span else None
    if ctx and ctx.is_valid:
        return trace.format_trace_id(ctx.trace_id)
    return None


def get_current_span_id() -> Optional[str]:
    """Return active span_id formatted as a 16-character hex string if active."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context() if current_span else None
    if ctx and ctx.is_valid:
        return trace.format_span_id(ctx.span_id)
    return None


def set_span_attributes(
    span: Optional[trace.Span],
    attributes: Dict[str, Any],
    record_payloads: Optional[bool] = None,
) -> None:
    """Set sanitized attributes onto an active span."""
    if span is None or not span.is_recording():
        return

    mgr = get_telemetry_manager()
    allow_payloads = record_payloads if record_payloads is not None else mgr.record_payloads
    clean = sanitize_attributes(attributes, record_payloads=allow_payloads)
    for k, v in clean.items():
        span.set_attribute(k, v)


def record_span_error(span: Optional[trace.Span], error: Any) -> None:
    """Safely record an exception or error onto an active span."""
    if span is None or not span.is_recording():
        return

    err_attrs = sanitize_error(error)
    span.set_status(Status(StatusCode.ERROR, err_attrs.get("error.message", "Operation failed")))
    for k, v in err_attrs.items():
        span.set_attribute(k, v)
    if isinstance(error, Exception):
        span.record_exception(error)


@asynccontextmanager
async def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_payloads: Optional[bool] = None,
) -> AsyncGenerator[Optional[trace.Span], None]:
    """Async context manager creating a traced span with automatic context propagation.

    Ensures that spans correctly link as children to any active span in the current
    execution context (including incoming HTTP requests instrumented by FastAPI).
    """
    mgr = get_telemetry_manager()
    if not mgr.is_enabled:
        yield None
        return

    allow_payloads = record_payloads if record_payloads is not None else mgr.record_payloads
    clean_attrs = sanitize_attributes(attributes or {}, record_payloads=allow_payloads)
    tracer = get_tracer()

    with tracer.start_as_current_span(name, attributes=clean_attrs) as span:
        try:
            yield span
        except Exception as exc:
            record_span_error(span, exc)
            raise


@contextmanager
def trace_span_sync(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_payloads: Optional[bool] = None,
) -> Generator[Optional[trace.Span], None, None]:
    """Synchronous context manager creating a traced span."""
    mgr = get_telemetry_manager()
    if not mgr.is_enabled:
        yield None
        return

    allow_payloads = record_payloads if record_payloads is not None else mgr.record_payloads
    clean_attrs = sanitize_attributes(attributes or {}, record_payloads=allow_payloads)
    tracer = get_tracer()

    with tracer.start_as_current_span(name, attributes=clean_attrs) as span:
        try:
            yield span
        except Exception as exc:
            record_span_error(span, exc)
            raise


def is_in_span(span_name: str) -> bool:
    """Check if the currently active span matches the given span name."""
    current = trace.get_current_span()
    return getattr(current, "name", None) == span_name


@asynccontextmanager
async def trace_tool_execution(
    tool_name: str,
    tool_type: str = "native",
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Optional[trace.Span], None]:
    """Context manager for tool execution that avoids creating duplicate nested tool spans."""
    if is_in_span("tool.execute"):
        current = trace.get_current_span()
        if extra_attributes:
            set_span_attributes(current, extra_attributes)
        yield current
        return

    attrs: Dict[str, Any] = {
        "tool_name": tool_name,
        "tool_type": tool_type,
    }
    if extra_attributes:
        attrs.update(extra_attributes)

    async with trace_span("tool.execute", attributes=attrs) as span:
        yield span
