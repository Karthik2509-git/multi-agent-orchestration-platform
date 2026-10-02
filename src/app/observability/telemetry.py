"""Centralized OpenTelemetry provider initialization and lifecycle management."""

from typing import List, Optional, Sequence

from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger

logger = get_logger(__name__)


class BoundedInMemorySpanExporter(InMemorySpanExporter):
    """In-memory span exporter enforcing a maximum capacity with FIFO eviction."""

    def __init__(self, max_spans: int = 500):
        super().__init__()
        self.max_spans = max_spans

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = super().export(spans)
        if len(self._finished_spans) > self.max_spans:
            overflow = len(self._finished_spans) - self.max_spans
            del self._finished_spans[:overflow]
        return result


class TelemetryManager:
    """Manages TracerProvider lifecycle, exporters, and in-memory span buffers."""

    def __init__(self) -> None:
        self._tracer_provider: Optional[TracerProvider] = None
        self._in_memory_exporter: Optional[BoundedInMemorySpanExporter] = None
        self._enabled: bool = False
        self._service_name: str = "multi-agent-orchestrator"
        self._record_payloads: bool = False

    @property
    def is_enabled(self) -> bool:
        """Return whether telemetry tracing is active."""
        return self._enabled

    @property
    def record_payloads(self) -> bool:
        """Return whether payload inspection is allowed in current environment."""
        return self._record_payloads

    @property
    def service_name(self) -> str:
        """Return configured service name."""
        return self._service_name

    def is_initialized(self) -> bool:
        """Return whether telemetry is initialized and actively enabled."""
        return self._enabled

    def initialize(self, settings: Optional[Settings] = None) -> None:
        """Initialize TracerProvider, span processors, and propagators."""
        app_settings = settings or get_settings()

        if not app_settings.telemetry_enabled:
            logger.info("Telemetry is disabled by configuration.")
            self._enabled = False
            return

        try:
            self._service_name = app_settings.telemetry_service_name
            self._record_payloads = app_settings.telemetry_record_payloads

            if self._tracer_provider is None:
                # Define OpenTelemetry service resource
                resource = Resource.create(
                    {
                        "service.name": app_settings.telemetry_service_name,
                        "service.version": app_settings.app_version,
                        "deployment.environment": app_settings.app_env,
                    }
                )

                provider = TracerProvider(resource=resource)

                # Configure exporter according to settings
                exporter_mode = app_settings.telemetry_exporter.lower()
                if exporter_mode == "memory":
                    self._in_memory_exporter = BoundedInMemorySpanExporter(
                        max_spans=app_settings.telemetry_max_in_memory_spans
                    )
                    provider.add_span_processor(SimpleSpanProcessor(self._in_memory_exporter))
                    logger.info(
                        "Initialized in-memory span exporter (max_spans=%d).",
                        app_settings.telemetry_max_in_memory_spans,
                    )
                elif exporter_mode == "console":
                    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                    logger.info("Initialized console span exporter.")
                elif exporter_mode == "none":
                    logger.info("Initialized TracerProvider with no active exporter.")

                # Register as global tracer provider
                trace.set_tracer_provider(provider)
                self._tracer_provider = provider

                # Register W3C TraceContext propagator
                set_global_textmap(TraceContextTextMapPropagator())

            self._enabled = True
            logger.info("OpenTelemetry tracing initialized for service '%s'.", self._service_name)
        except Exception as e:
            logger.warning(
                "Failed to initialize OpenTelemetry tracing: %s. Continuing without telemetry.", e
            )
            self._enabled = False

    def shutdown(self) -> None:
        """Flush and reset active telemetry state."""
        if self._in_memory_exporter:
            self._in_memory_exporter.clear()
        self._enabled = False
        logger.info("Telemetry manager shutdown complete.")

    def get_tracer(self, name: str = "multi_agent_orchestrator") -> trace.Tracer:
        """Return a tracer instance bound to the configured provider."""
        if self._tracer_provider:
            return self._tracer_provider.get_tracer(name)
        return trace.get_tracer(name)

    def get_in_memory_spans(self) -> List[ReadableSpan]:
        """Return all finished spans currently buffered in memory."""
        if self._in_memory_exporter:
            return list(self._in_memory_exporter.get_finished_spans())
        return []

    def clear_in_memory_spans(self) -> None:
        """Clear finished spans from the in-memory buffer."""
        if self._in_memory_exporter:
            self._in_memory_exporter.clear()

    def get_tracer_provider(self) -> Optional[TracerProvider]:
        """Return active TracerProvider instance."""
        return self._tracer_provider


# Global singleton instance
_telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    """Return the global TelemetryManager instance."""
    return _telemetry_manager


def init_telemetry(settings: Optional[Settings] = None) -> TelemetryManager:
    """Initialize telemetry subsystem with given settings."""
    _telemetry_manager.initialize(settings)
    return _telemetry_manager


def shutdown_telemetry() -> None:
    """Shut down global telemetry subsystem."""
    _telemetry_manager.shutdown()


def get_tracer(name: str = "multi_agent_orchestrator") -> trace.Tracer:
    """Convenience helper returning the active tracer."""
    return _telemetry_manager.get_tracer(name)
