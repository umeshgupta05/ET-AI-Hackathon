"""Optional OpenTelemetry wiring for production deployments."""

from __future__ import annotations

import os


def configure_telemetry(app) -> dict:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    service_name = os.getenv("OTEL_SERVICE_NAME", "digital-public-safety-shield")
    if not endpoint:
        return {"enabled": False, "status": "disabled"}
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        return {"enabled": True, "status": "ready", "endpoint": endpoint, "service_name": service_name}
    except Exception as exc:
        return {"enabled": True, "status": "unavailable", "detail": str(exc)}
