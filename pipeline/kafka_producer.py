"""
Kafka producer – reads synthetic transactions from the FastAPI data API,
serialises them with Protobuf and publishes to the ``raw-transactions`` topic.

Usage
-----
    python pipeline/kafka_producer.py \
        --api-url http://localhost:8000 \
        --kafka-brokers localhost:9092 \
        --batch-size 50 \
        --interval 5
"""

import argparse
import logging
import os
import struct
import time
from pathlib import Path
from typing import Optional

import requests
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOPIC = "raw-transactions"


# ── Protobuf serialisation ─────────────────────────────────────────────────

def _load_proto():
    """Return the Protobuf Transaction class if generated bindings exist."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "schemas" / "generated"))
        from transaction_pb2 import Transaction, MerchantCategory, DeviceType  # noqa
        return Transaction, MerchantCategory, DeviceType
    except ImportError:
        return None, None, None


_ProtoTransaction, _MerchantCategory, _DeviceType = _load_proto()

_MERCHANT_STR_TO_ENUM = {
    "electronics": 1, "grocery": 2, "travel": 3, "dining": 4,
    "entertainment": 5, "fashion": 6, "health": 7,
}
_DEVICE_STR_TO_ENUM = {
    "mobile": 1, "desktop": 2, "tablet": 3, "pos_terminal": 4,
}


def serialise(record: dict) -> bytes:
    """Serialise a transaction dict to bytes (Protobuf if available, else JSON)."""
    if _ProtoTransaction is not None:
        msg = _ProtoTransaction()
        msg.customer_id = record["customer_id"]
        msg.transaction_amount = float(record["transaction_amount"])
        msg.merchant_category = _MERCHANT_STR_TO_ENUM.get(record["merchant_category"], 0)
        msg.device_type = _DEVICE_STR_TO_ENUM.get(record["device_type"], 0)
        msg.timestamp = int(record["timestamp"])
        return msg.SerializeToString()

    # Fallback: JSON bytes
    import json
    return json.dumps(record).encode()


# ── Kafka helpers ─────────────────────────────────────────────────────────

def ensure_topic(brokers: str, topic: str, partitions: int = 3, replication: int = 1) -> None:
    admin = AdminClient({"bootstrap.servers": brokers})
    metadata = admin.list_topics(timeout=5)
    if topic not in metadata.topics:
        new_topic = NewTopic(topic, num_partitions=partitions, replication_factor=replication)
        fs = admin.create_topics([new_topic])
        for t, f in fs.items():
            try:
                f.result()
                logger.info("Topic '%s' created.", t)
            except Exception as e:
                logger.warning("Topic creation: %s", e)


def delivery_report(err, msg):
    if err:
        logger.error("Delivery failed for key %s: %s", msg.key(), err)


# ── Main ──────────────────────────────────────────────────────────────────

def run_producer(
    api_url: str,
    brokers: str,
    batch_size: int,
    interval: float,
    max_batches: Optional[int] = None,
) -> None:
    ensure_topic(brokers, TOPIC)
    producer = Producer({"bootstrap.servers": brokers, "acks": "all"})

    batch_count = 0
    while max_batches is None or batch_count < max_batches:
        try:
            resp = requests.get(
                f"{api_url.rstrip('/')}/transactions?batch_size={batch_size}", timeout=10
            )
            resp.raise_for_status()
            records = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch transactions: %s", exc)
            time.sleep(interval)
            continue

        for rec in records:
            payload = serialise(rec)
            producer.produce(
                TOPIC,
                key=rec["customer_id"].encode(),
                value=payload,
                callback=delivery_report,
            )

        producer.poll(0)
        batch_count += 1
        logger.info("Published batch %d (%d records).", batch_count, len(records))
        time.sleep(interval)

    producer.flush()
    logger.info("Producer finished.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Kafka transaction producer")
    parser.add_argument("--api-url", default=os.getenv("DATA_API_URL", "http://localhost:8000"))
    parser.add_argument("--kafka-brokers", default=os.getenv("KAFKA_BROKERS", "localhost:9092"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_producer(args.api_url, args.kafka_brokers, args.batch_size, args.interval, args.max_batches)
