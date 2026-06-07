# Quick ML Classifier – End-to-End Fraud Detection System

A production-grade, end-to-end ML pipeline for real-time fraud detection.

```
quick-ml-classifier/
├── services/data_api/          # FastAPI synthetic transaction + label generator
├── schemas/                    # Protobuf schema + compile script
├── ingestion/                  # Parquet ingestion pipeline (PyArrow)
├── pipeline/                   # Kafka producer/consumer + Medallion (Bronze/Silver/Gold)
├── features/                   # PySpark feature engineering → Iceberg
├── training/                   # LightGBM + Optuna hyperparameter tuning + MLflow
├── registry/                   # MLflow model registry + Confluent Schema Registry
├── deployment/                 # FastAPI inference server + SageMaker deploy script
├── observability/              # Prometheus, Grafana dashboard, OTel Collector, Jaeger
├── docker-compose.yml          # Full local stack
└── requirements.txt
```

---

## Architecture Overview

```
FastAPI Data API  ──►  Kafka (raw-transactions)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           Bronze          Silver          Gold
        (Iceberg)  ──►  (Iceberg)  ──►  (Iceberg)
                                            │
                                     PySpark Features
                                            │
                                     LightGBM Training
                                            │
                                     MLflow Registry
                                            │
                                    SageMaker Endpoint
                                            │
                            Prometheus / Grafana / OTel
```

---

## Quick Start

### 1. Start the full local stack

```bash
docker compose up --build
```

| Service              | URL                        |
|----------------------|----------------------------|
| Data API             | http://localhost:8000/docs |
| Kafka                | localhost:29092             |
| Schema Registry      | http://localhost:8085      |
| MinIO console        | http://localhost:9001      |
| Iceberg REST catalog | http://localhost:8181      |
| MLflow               | http://localhost:5000      |
| Inference service    | http://localhost:8080/docs |
| Prometheus           | http://localhost:9090      |
| Grafana              | http://localhost:3000      |
| Jaeger               | http://localhost:16686     |

Default MinIO credentials: `minioadmin / minioadmin`
Default Grafana credentials: `admin / admin`

---

### 2. Compile Protobuf schema

```bash
bash schemas/compile_proto.sh
```

### 3. Run the Parquet ingestion pipeline (standalone)

```bash
pip install -r requirements.txt
python ingestion/ingest.py \
    --api-url http://localhost:8000 \
    --batch-size 100 \
    --num-batches 20 \
    --output-dir data/parquet
```

### 4. Register the Protobuf schema in Schema Registry

```bash
python registry/schema_registry.py \
    --action register \
    --registry-url http://localhost:8085 \
    --proto-path schemas/transaction.proto
```

### 5. Run the PySpark feature engineering pipeline

```bash
spark-submit \
  --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0,org.apache.hadoop:hadoop-aws:3.3.4 \
  features/spark_pipeline.py \
  --catalog-uri http://localhost:8181 \
  --s3-endpoint http://localhost:9000
```

### 6. Train the LightGBM classifier

```bash
python training/train.py \
    --data-path data/parquet \
    --mlflow-uri http://localhost:5000 \
    --n-trials 30
```

Training automatically:
- Runs Optuna hyperparameter search optimising `average_precision_score`
- Sets `scale_pos_weight` to handle class imbalance
- Logs every trial as an MLflow nested run
- Registers the best pipeline as `fraud-classifier` and transitions it to **Production**

### 7. Promote the best model

```bash
python registry/model_registry.py \
    --action promote \
    --mlflow-uri http://localhost:5000
```

### 8. Deploy to Amazon SageMaker

```bash
# Build inference image
docker build -t fraud-classifier:latest -f deployment/Dockerfile .

# Upload model to S3
tar -czf model.tar.gz -C models fraud_classifier.pkl
aws s3 cp model.tar.gz s3://<YOUR_BUCKET>/fraud-classifier/model.tar.gz

# Deploy
export AWS_ACCOUNT_ID=<your-account-id>
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/SageMakerExecutionRole
export S3_MODEL_ARTIFACT_URI=s3://<YOUR_BUCKET>/fraud-classifier/model.tar.gz

python deployment/sagemaker_deploy.py
```

### 9. Call the inference endpoint

```bash
# Local
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "1234",
    "transaction_amount": 3500.0,
    "merchant_category": "electronics",
    "device_type": "mobile",
    "timestamp": 1749200000
  }'

# SageMaker
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name fraud-classifier-endpoint \
  --content-type application/json \
  --body '{"customer_id":"1234","transaction_amount":3500.0,"merchant_category":"electronics","device_type":"mobile","timestamp":1749200000}' \
  /tmp/response.json
```

---

## Component Details

### Data API (`services/data_api/`)
- `GET /transaction` – single random transaction
- `GET /transactions?batch_size=N` – batch of N transactions
- `GET /labels` – fraud/non_fraud labels (~7 % fraud rate)
- `GET /label/{customer_id}` – label for a specific customer
- `GET /health` – liveness probe

### Protobuf Schema (`schemas/transaction.proto`)
- Defines `Transaction`, `TransactionLabel`, `TransactionBatch` messages
- `MerchantCategory` and `DeviceType` enums
- New optional fields can be added without breaking existing consumers (BACKWARD compatibility)

### Kafka Pipeline (`pipeline/`)
- **Producer**: polls Data API → serialises with Protobuf → publishes to `raw-transactions`
- **Consumer**: Bronze (raw append) → Silver (dedup + label join) → Gold (1h/24h rolling aggregation), all backed by Apache Iceberg

### PySpark Features (`features/spark_pipeline.py`)
Computed features:
| Feature | Description |
|---|---|
| `transaction_amount` | Raw amount |
| `transaction_amount_scaled` | StandardScaler output |
| `hour_of_day` | 0–23 |
| `day_of_week` | 0 (Mon) – 6 (Sun) |
| `merchant_category_encoded` | Ordinal via `StringIndexer` |
| `device_type_encoded` | Ordinal via `StringIndexer` |
| `txn_count_1h` | Rolling count (1-hour window) |
| `total_amount_1h` | Rolling sum (1-hour window) |
| `txn_count_24h` | Rolling count (24-hour window) |
| `total_amount_24h` | Rolling sum (24-hour window) |

### Training (`training/train.py`)
- **Model**: `sklearn.Pipeline` → `OrdinalEncoder` + `LGBMClassifier`
- **Imbalance**: `scale_pos_weight = n_negative / n_positive`
- **Metric**: `average_precision_score` (area under PR curve)
- **Tuning**: Optuna, 30 trials by default
- **Hyperparams**: `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

### Registries
- **Model Registry**: MLflow – every run logged; best run promoted to `Production`
- **Schema Registry**: Confluent – BACKWARD compatibility enforced on `transactions-value` subject

### Observability
| Tool | Purpose |
|---|---|
| Prometheus | Metrics scraping |
| Grafana | Dashboards (pre-loaded) |
| OpenTelemetry Collector | Span & metric collection |
| Jaeger | Distributed trace UI |

Key metrics: `fraud_predictions_total`, `prediction_latency_seconds`, `model_version_info`, `kafka_messages_total`, `ingestion_records_total`