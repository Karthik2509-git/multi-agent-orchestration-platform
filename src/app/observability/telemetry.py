"""Centralized OpenTelemetry provider initialization and lifecycle management."""

from typing import Any, List, Optional, Sequence

from opentelemetry import metrics, trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    InMemoryMetricReader,
    MetricsData,
    PeriodicExportingMetricReader,
)
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
    """Manages TracerProvider and MeterProvider lifecycle, exporters, and in-memory buffers."""

    def __init__(self) -> None:
        self._tracer_provider: Optional[TracerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._in_memory_exporter: Optional[BoundedInMemorySpanExporter] = None
        self._in_memory_metric_reader: Optional[InMemoryMetricReader] = None
        self._enabled: bool = False
        self._service_name: str = "multi-agent-orchestrator"
        self._record_payloads: bool = False

    @property
    def is_enabled(self) -> bool:
        """Return whether telemetry tracing and metrics are active."""
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
        """Initialize TracerProvider, MeterProvider, processors, and propagators."""
        app_settings = settings or get_settings()

        if not app_settings.telemetry_enabled:
            logger.info("Telemetry is disabled by configuration.")
            self._enabled = False
            return

        try:
            self._service_name = app_settings.telemetry_service_name
            self._record_payloads = app_settings.telemetry_record_payloads
            app_env = getattr(app_settings, "app_env", "development") or "development"

            resource = Resource.create(
                {
                    "service.name": app_settings.telemetry_service_name,
                    "service.version": app_settings.app_version,
                    "deployment.environment": app_env,
                }
            )

            exporter_mode = app_settings.telemetry_exporter.lower()

            if self._tracer_provider is None:
                provider = TracerProvider(resource=resource)

                # Configure trace exporter according to settings
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

            if self._meter_provider is None:
                metric_readers = []
                if exporter_mode == "memory":
                    self._in_memory_metric_reader = InMemoryMetricReader()
                    metric_readers.append(self._in_memory_metric_reader)
                    logger.info("Initialized in-memory metric reader.")
                elif exporter_mode == "console":
                    metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
                    logger.info("Initialized console metric exporter.")
                elif exporter_mode == "none":
                    logger.info("Initialized MeterProvider with no active metric reader.")

                meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
                metrics.set_meter_provider(meter_provider)
                self._meter_provider = meter_provider

            self._enabled = True
            logger.info(
                "OpenTelemetry tracing and metrics initialized for service '%s'.",
                self._service_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize OpenTelemetry telemetry: %s. Continuing without telemetry.", e
            )
            self._enabled = False

    def shutdown(self) -> None:
        """Flush and reset active telemetry state."""
        if self._in_memory_exporter:
            self._in_memory_exporter.clear()
        if self._meter_provider:
            try:
                self._meter_provider.shutdown()
            except Exception:
                pass
            self._meter_provider = None
        self._in_memory_metric_reader = None
        self._enabled = False
        logger.info("Telemetry manager shutdown complete.")

    def get_tracer(self, name: str = "multi_agent_orchestrator") -> trace.Tracer:
        """Return a tracer instance bound to the configured provider."""
        if self._tracer_provider:
            return self._tracer_provider.get_tracer(name)
        return trace.get_tracer(name)

    def get_meter(self, name: str = "multi_agent_orchestrator") -> metrics.Meter:
        """Return a meter instance bound to the configured provider."""
        if self._meter_provider:
            return self._meter_provider.get_meter(name)
        return metrics.get_meter(name)

    def get_in_memory_spans(self) -> List[ReadableSpan]:
        """Return all finished spans currently buffered in memory."""
        if self._in_memory_exporter:
            return list(self._in_memory_exporter.get_finished_spans())
        return []

    def clear_in_memory_spans(self) -> None:
        """Clear finished spans from the in-memory buffer."""
        if self._in_memory_exporter:
            self._in_memory_exporter.clear()

    def get_in_memory_metrics(self) -> Optional[MetricsData]:
        """Return collected metrics data from the in-memory reader."""
        if self._in_memory_metric_reader:
            return self._in_memory_metric_reader.get_metrics_data()
        return None

    def get_metric_data_points(self, metric_name: str) -> List[Any]:
        """Helper to extract data points for a specific metric name from in-memory metrics."""
        data = self.get_in_memory_metrics()
        if not data:
            return []
        data_points = []
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    if m.name == metric_name and hasattr(m.data, "data_points"):
                        data_points.extend(m.data.data_points)
        return data_points

    def reset_metrics(self) -> None:
        """Reset the meter provider and in-memory metric reader for test isolation."""
        if self._enabled:
            resource = Resource.create({"service.name": self._service_name})
            self._in_memory_metric_reader = InMemoryMetricReader()
            self._meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[self._in_memory_metric_reader],
            )

    def get_tracer_provider(self) -> Optional[TracerProvider]:
        """Return active TracerProvider instance."""
        return self._tracer_provider

    def get_meter_provider(self) -> Optional[MeterProvider]:
        """Return active MeterProvider instance."""
        return self._meter_provider


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


def get_meter(name: str = "multi_agent_orchestrator") -> metrics.Meter:
    """Convenience helper returning the active meter."""
    return _telemetry_manager.get_meter(name)


def get_in_memory_metrics() -> Optional[MetricsData]:
    """Convenience helper returning collected metrics data."""
    return _telemetry_manager.get_in_memory_metrics()


def reset_metrics() -> None:
    """Convenience helper resetting active metrics for test isolation."""
    _telemetry_manager.reset_metrics()


def get_tracer_provider() -> Optional[TracerProvider]:
    """Convenience helper returning the active TracerProvider."""
    return _telemetry_manager.get_tracer_provider()


def get_meter_provider() -> Optional[MeterProvider]:
    """Convenience helper returning the active MeterProvider."""
    return _telemetry_manager.get_meter_provider()
