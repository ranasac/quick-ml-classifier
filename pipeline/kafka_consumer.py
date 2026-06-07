"""
Kafka consumer – reads raw Protobuf messages from ``raw-transactions`` and
fans them out across the three Medallion layers backed by Apache Iceberg tables
(via PyIceberg + a local REST catalog backed by MinIO).

Medallion layers
----------------
bronze  – raw deserialized records, append-only
silver  – deduplicated + type-validated + label-joined
gold    – aggregated feature rows (rolling windows)

Usage
-----
    python pipeline/kafka_consumer.py \
        --kafka-brokers localhost:9092 \
        --catalog-uri http://localhost:8181 \
        --api-url http://localhost:8000
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pyarrow as pa
import requests
from confluent_kafka import Consumer, KafkaError, KafkaException
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOPIC = "raw-transactions"

# ── Iceberg schemas ───────────────────────────────────────────────────────────

BRONZE_SCHEMA = Schema(
    NestedField(1, "customer_id", StringType(), required=True),
    NestedField(2, "transaction_amount", DoubleType(), required=True),
    NestedField(3, "merchant_category", StringType(), required=False),
    NestedField(4, "device_type", StringType(), required=False),
    NestedField(5, "timestamp", LongType(), required=True),
    NestedField(6, "ingest_time", LongType(), required=False),
)

SILVER_SCHEMA = Schema(
    NestedField(1, "customer_id", StringType(), required=True),
    NestedField(2, "transaction_amount", DoubleType(), required=True),
    NestedField(3, "merchant_category", StringType(), required=False),
    NestedField(4, "device_type", StringType(), required=False),
    NestedField(5, "timestamp", LongType(), required=True),
    NestedField(6, "label", StringType(), required=False),
    NestedField(7, "ingest_time", LongType(), required=False),
)

GOLD_SCHEMA = Schema(
    NestedField(1, "customer_id", StringType(), required=True),
    NestedField(2, "window_start", LongType(), required=True),
    NestedField(3, "window_end", LongType(), required=True),
    NestedField(4, "txn_count", LongType(), required=True),
    NestedField(5, "total_amount", DoubleType(), required=True),
    NestedField(6, "max_amount", DoubleType(), required=True),
    NestedField(7, "avg_amount", DoubleType(), required=True),
    NestedField(8, "label", StringType(), required=False),
    NestedField(9, "computed_at", LongType(), required=False),
)

# ── Arrow schemas matching Iceberg ────────────────────────────────────────────

BRONZE_ARROW = pa.schema([
    ("customer_id", pa.string()),
    ("transaction_amount", pa.float64()),
    ("merchant_category", pa.string()),
    ("device_type", pa.string()),
    ("timestamp", pa.int64()),
    ("ingest_time", pa.int64()),
])

SILVER_ARROW = pa.schema([
    ("customer_id", pa.string()),
    ("transaction_amount", pa.float64()),
    ("merchant_category", pa.string()),
    ("device_type", pa.string()),
    ("timestamp", pa.int64()),
    ("label", pa.string()),
    ("ingest_time", pa.int64()),
])

GOLD_ARROW = pa.schema([
    ("customer_id", pa.string()),
    ("window_start", pa.int64()),
    ("window_end", pa.int64()),
    ("txn_count", pa.int64()),
    ("total_amount", pa.float64()),
    ("max_amount", pa.float64()),
    ("avg_amount", pa.float64()),
    ("label", pa.string()),
    ("computed_at", pa.int64()),
])


# ── Protobuf deserialisation ──────────────────────────────────────────────────

def _load_proto():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "schemas" / "generated"))
        from transaction_pb2 import Transaction, MerchantCategory, DeviceType
        _MERCHANT = {v: k.lower() for k, v in MerchantCategory.items() if v != 0}
        _DEVICE = {v: k.lower() for k, v in DeviceType.items() if v != 0}
        return Transaction, _MERCHANT, _DEVICE
    except ImportError:
        return None, None, None


_ProtoTx, _MERCHANT_MAP, _DEVICE_MAP = _load_proto()


def deserialise(data: bytes) -> Optional[Dict[str, Any]]:
    """Deserialise Protobuf bytes; fall back to JSON."""
    if _ProtoTx is not None:
        try:
            msg = _ProtoTx()
            msg.ParseFromString(data)
            return {
                "customer_id": msg.customer_id,
                "transaction_amount": msg.transaction_amount,
                "merchant_category": _MERCHANT_MAP.get(msg.merchant_category, "unknown"),
                "device_type": _DEVICE_MAP.get(msg.device_type, "unknown"),
                "timestamp": msg.timestamp,
            }
        except Exception:
            pass
    try:
        return json.loads(data.decode())
    except Exception:
        return None


# ── Iceberg helpers ───────────────────────────────────────────────────────────

def get_or_create_table(catalog, namespace: str, table_name: str, schema: Schema):
    full_name = f"{namespace}.{table_name}"
    try:
        return catalog.load_table(full_name)
    except NoSuchTableError:
        catalog.create_namespace_if_not_exists(namespace)
        return catalog.create_table(full_name, schema=schema)


def append_records(table, records: List[Dict[str, Any]], arrow_schema: pa.Schema) -> None:
    if not records:
        return
    arrow_table = pa.Table.from_pylist(records, schema=arrow_schema)
    table.append(arrow_table)
    logger.debug("Appended %d records to %s", len(records), table.name())


# ── Label cache ───────────────────────────────────────────────────────────────

class LabelCache:
    def __init__(self, api_url: str, ttl: int = 300):
        self.api_url = api_url.rstrip("/")
        self.ttl = ttl
        self._cache: Dict[str, str] = {}
        self._last_refresh = 0.0

    def get(self, customer_id: str) -> str:
        if time.time() - self._last_refresh > self.ttl:
            self._refresh()
        return self._cache.get(customer_id, "non_fraud")

    def _refresh(self):
        try:
            resp = requests.get(f"{self.api_url}/labels", timeout=10)
            resp.raise_for_status()
            for item in resp.json():
                self._cache[item["customer_id"]] = item["label"]
            self._last_refresh = time.time()
            logger.info("Label cache refreshed (%d entries).", len(self._cache))
        except Exception as exc:
            logger.warning("Could not refresh label cache: %s", exc)


# ── Gold aggregation (in-memory micro-batch, 1-hour window) ──────────────────

def aggregate_gold(records: List[Dict[str, Any]], label_cache: LabelCache) -> List[Dict[str, Any]]:
    from collections import defaultdict
    WINDOW = 3600  # 1-hour window
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    for r in records:
        t = r["timestamp"]
        window_start = (t // WINDOW) * WINDOW
        key = (r["customer_id"], window_start)
        buckets[key].append(r["transaction_amount"])

    now = int(time.time())
    rows = []
    for (cid, ws), amounts in buckets.items():
        rows.append({
            "customer_id": cid,
            "window_start": ws,
            "window_end": ws + WINDOW,
            "txn_count": len(amounts),
            "total_amount": sum(amounts),
            "max_amount": max(amounts),
            "avg_amount": sum(amounts) / len(amounts),
            "label": label_cache.get(cid),
            "computed_at": now,
        })
    return rows


# ── Consumer main loop ────────────────────────────────────────────────────────

def run_consumer(
    brokers: str,
    catalog_uri: str,
    api_url: str,
    group_id: str = "medallion-consumer",
    batch_size: int = 100,
    flush_interval: float = 10.0,
) -> None:
    # Build catalog
    catalog = load_catalog(
        "default",
        **{
            "uri": catalog_uri,
            "s3.endpoint": os.getenv("S3_ENDPOINT", "http://localhost:9000"),
            "s3.access-key-id": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            "s3.secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        },
    )

    bronze_tbl = get_or_create_table(catalog, "bronze", "transactions", BRONZE_SCHEMA)
    silver_tbl = get_or_create_table(catalog, "silver", "transactions_labeled", SILVER_SCHEMA)
    gold_tbl = get_or_create_table(catalog, "gold", "features", GOLD_SCHEMA)

    label_cache = LabelCache(api_url)

    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([TOPIC])

    buffer: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    last_flush = time.time()

    logger.info("Consumer started. Listening on '%s'…", TOPIC)
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                pass
            elif msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(msg.error())
            else:
                record = deserialise(msg.value())
                if record:
                    buffer.append(record)
                consumer.commit(asynchronous=False)

            # Flush buffer every N records or every flush_interval seconds
            if len(buffer) >= batch_size or (buffer and time.time() - last_flush >= flush_interval):
                now = int(time.time())

                # Bronze – raw append
                bronze_rows = [{**r, "ingest_time": now} for r in buffer]
                append_records(bronze_tbl, bronze_rows, BRONZE_ARROW)

                # Silver – deduplicate by (customer_id, timestamp)
                silver_rows = []
                for r in buffer:
                    key = f"{r['customer_id']}_{r['timestamp']}"
                    if key not in seen_ids:
                        seen_ids.add(key)
                        silver_rows.append({
                            **r,
                            "label": label_cache.get(r["customer_id"]),
                            "ingest_time": now,
                        })
                append_records(silver_tbl, silver_rows, SILVER_ARROW)

                # Gold – 1-hour rolling aggregation
                gold_rows = aggregate_gold(buffer, label_cache)
                append_records(gold_tbl, gold_rows, GOLD_ARROW)

                logger.info(
                    "Flushed: bronze=%d silver=%d gold=%d",
                    len(bronze_rows), len(silver_rows), len(gold_rows),
                )
                buffer.clear()
                last_flush = time.time()

    except KeyboardInterrupt:
        logger.info("Shutting down consumer.")
    finally:
        consumer.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Kafka Medallion consumer")
    parser.add_argument("--kafka-brokers", default=os.getenv("KAFKA_BROKERS", "localhost:9092"))
    parser.add_argument("--catalog-uri", default=os.getenv("ICEBERG_CATALOG_URI", "http://localhost:8181"))
    parser.add_argument("--api-url", default=os.getenv("DATA_API_URL", "http://localhost:8000"))
    parser.add_argument("--group-id", default="medallion-consumer")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--flush-interval", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_consumer(
        brokers=args.kafka_brokers,
        catalog_uri=args.catalog_uri,
        api_url=args.api_url,
        group_id=args.group_id,
        batch_size=args.batch_size,
        flush_interval=args.flush_interval,
    )
