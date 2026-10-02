"""Observability, OpenTelemetry tracing, and telemetry redaction."""

from src.app.observability.spans import (
    get_current_span_id,
    get_current_trace_id,
    is_in_span,
    record_span_error,
    set_span_attributes,
    trace_span,
    trace_span_sync,
    trace_tool_execution,
)
from src.app.observability.telemetry import (
    BoundedInMemorySpanExporter,
    TelemetryManager,
    get_telemetry_manager,
    get_tracer,
    init_telemetry,
    shutdown_telemetry,
)

__all__ = [
    "BoundedInMemorySpanExporter",
    "TelemetryManager",
    "get_current_span_id",
    "get_current_trace_id",
    "get_telemetry_manager",
    "get_tracer",
    "init_telemetry",
    "is_in_span",
    "record_span_error",
    "set_span_attributes",
    "shutdown_telemetry",
    "trace_span",
    "trace_span_sync",
    "trace_tool_execution",
]
