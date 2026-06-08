"""
OpenTelemetry + Prometheus configuration for the fraud detection inference service.

This module is imported by deployment/serve.py and any other FastAPI service
that needs observability.

Metrics tracked
---------------
fraud_predictions_total        – counter, labelled by predicted_label
prediction_latency_seconds     – histogram
request_count_total            – counter (provided by prometheus-fastapi-instrumentator)
model_version                  – gauge (the current deployed model version string)
"""

import logging
import os
import time
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total number of fraud / non_fraud predictions",
    labelnames=["predicted_label"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to produce a single prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently loaded model version (string label encoded as 1.0)",
    labelnames=["version"],
)

INGESTION_RECORDS = Counter(
    "ingestion_records_total",
    "Total records ingested into the Parquet / Iceberg pipeline",
    labelnames=["layer"],  # bronze | silver | gold
)

KAFKA_MESSAGES = Counter(
    "kafka_messages_total",
    "Total Kafka messages produced / consumed",
    labelnames=["direction", "topic"],  # produced | consumed
)


def record_prediction(label: str, latency_seconds: float) -> None:
    """Record a single prediction outcome and its latency."""
    FRAUD_PREDICTIONS.labels(predicted_label=label).inc()
    PREDICTION_LATENCY.observe(latency_seconds)


def set_model_version(version: str) -> None:
    MODEL_VERSION.labels(version=version).set(1.0)


def timed_prediction(func: Callable) -> Callable:
    """Decorator that wraps a prediction function and records its latency."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency = time.perf_counter() - start
        # result is expected to be a list of PredictionOutput
        for pred in result if isinstance(result, list) else [result]:
            record_prediction(pred.label, latency / max(len(result), 1))
        return result

    return wrapper


# ── OpenTelemetry setup ───────────────────────────────────────────────────────

def setup_otel(service_name: str = "fraud-classifier") -> None:
    """
    Configure OpenTelemetry tracing and export spans to an OTLP endpoint.
    Set OTEL_EXPORTER_OTLP_ENDPOINT to point at a Jaeger / OTel Collector.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
        trace.set_tracer_provider(provider)
        logger.info("OTel tracing configured → %s", otlp_endpoint)
    except ImportError:
        logger.warning("opentelemetry packages not installed; tracing disabled.")


def start_prometheus_server(port: int = 9090) -> None:
    """Start a standalone Prometheus metrics HTTP server (non-FastAPI services)."""
    start_http_server(port)
    logger.info("Prometheus metrics server started on :%d", port)
