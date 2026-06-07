"""
SageMaker inference server.

Exposes two endpoints required by the SageMaker real-time inference contract:
  GET  /ping          – liveness / health check
  POST /invocations   – model inference

The model is loaded at startup from:
  1. The path given in MODEL_PATH env var (local file, e.g. /opt/ml/model/fraud_classifier.pkl)
  2. An MLflow model URI given in MLFLOW_MODEL_URI env var
  3. A fallback path: /opt/ml/model/fraud_classifier.pkl

Input JSON (single transaction)
--------------------------------
{
  "customer_id": "1234",
  "transaction_amount": 350.5,
  "merchant_category": "electronics",
  "device_type": "mobile",
  "timestamp": 1749200000
}

Input JSON (batch)
------------------
[{...}, {...}]

Output
------
{
  "predictions": [{"customer_id": "1234", "score": 0.87, "label": "fraud"}]
}
"""

import logging
import os
import time
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

# ── Optional OpenTelemetry / Prometheus instrumentation ──────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "merchant_category",
    "device_type",
    "transaction_amount",
    "hour_of_day",
    "day_of_week",
    "txn_count_1h",
    "total_amount_1h",
    "txn_count_24h",
    "total_amount_24h",
]

# ── Load model ────────────────────────────────────────────────────────────────

def _load_model():
    mlflow_uri = os.getenv("MLFLOW_MODEL_URI", "")
    model_path = os.getenv("MODEL_PATH", "/opt/ml/model/fraud_classifier.pkl")

    if mlflow_uri:
        import mlflow.sklearn
        logger.info("Loading model from MLflow URI: %s", mlflow_uri)
        return mlflow.sklearn.load_model(mlflow_uri)

    if os.path.exists(model_path):
        logger.info("Loading model from %s", model_path)
        return joblib.load(model_path)

    raise RuntimeError(
        "No model found. Set MODEL_PATH or MLFLOW_MODEL_URI environment variable."
    )


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Fraud Classifier Inference Service", version="1.0.0")

# Prometheus metrics
if _PROM_AVAILABLE:
    Instrumentator().instrument(app).expose(app)

# OpenTelemetry tracing
if _OTEL_AVAILABLE:
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    resource = Resource.create({"service.name": "fraud-classifier"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


# ── Pydantic models ───────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    customer_id: str
    transaction_amount: float = Field(..., ge=0)
    merchant_category: str = "unknown"
    device_type: str = "unknown"
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    # Rolling features (optional – default to 0 when not supplied)
    txn_count_1h: int = 1
    total_amount_1h: float = 0.0
    txn_count_24h: int = 1
    total_amount_24h: float = 0.0


class PredictionOutput(BaseModel):
    customer_id: str
    score: float
    label: str


class PredictionResponse(BaseModel):
    predictions: List[PredictionOutput]


# ── Helper ────────────────────────────────────────────────────────────────────

def _enrich(txn: TransactionInput) -> Dict[str, Any]:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(txn.timestamp, tz=timezone.utc)
    return {
        "merchant_category": txn.merchant_category,
        "device_type": txn.device_type,
        "transaction_amount": txn.transaction_amount,
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "txn_count_1h": txn.txn_count_1h,
        "total_amount_1h": txn.total_amount_1h,
        "txn_count_24h": txn.txn_count_24h,
        "total_amount_24h": txn.total_amount_24h,
    }


def _predict(transactions: List[TransactionInput]) -> List[PredictionOutput]:
    model = get_model()
    rows = [_enrich(t) for t in transactions]
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    scores = model.predict_proba(df)[:, 1]
    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    results = []
    for txn, score in zip(transactions, scores):
        results.append(
            PredictionOutput(
                customer_id=txn.customer_id,
                score=round(float(score), 6),
                label="fraud" if score >= threshold else "non_fraud",
            )
        )
    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    """SageMaker liveness probe."""
    try:
        get_model()
        return Response(content="", status_code=200)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/invocations", response_model=PredictionResponse)
async def invocations(request: Request):
    """SageMaker inference endpoint – accepts single or batch JSON."""
    body = await request.json()

    if isinstance(body, dict):
        transactions = [TransactionInput(**body)]
    elif isinstance(body, list):
        transactions = [TransactionInput(**item) for item in body]
    else:
        raise HTTPException(status_code=400, detail="Request body must be a JSON object or array.")

    predictions = _predict(transactions)
    return PredictionResponse(predictions=predictions)
