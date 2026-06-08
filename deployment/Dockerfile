# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY deployment/requirements-serve.txt requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY deployment/serve.py serve.py

# ── Model artifact (copy from build context if pre-trained locally) ───────────
# In production, the model is loaded from S3/MLflow at runtime via env vars.
# Uncomment the next line to bake a local model into the image:
# COPY models/fraud_classifier.pkl /opt/ml/model/fraud_classifier.pkl

# ── Runtime config ────────────────────────────────────────────────────────────
ENV MODEL_PATH=/opt/ml/model/fraud_classifier.pkl
ENV FRAUD_THRESHOLD=0.5
ENV PORT=8080

EXPOSE 8080

# SageMaker calls the container with "serve" as the command
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
