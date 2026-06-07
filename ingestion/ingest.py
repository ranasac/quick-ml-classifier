"""
Parquet ingestion pipeline.

Steps
-----
1. Poll the FastAPI /transactions endpoint in batches.
2. Parse each response using the generated Protobuf schema (falls back to
   plain-dict parsing when protoc-generated code is not available).
3. Write Parquet files partitioned by ``date`` (derived from timestamp) and
   sorted by ``customer_id`` using PyArrow.

Usage
-----
    python ingestion/ingest.py --api-url http://localhost:8000 \
        --batch-size 100 --num-batches 10 --output-dir data/parquet
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Arrow schema ──────────────────────────────────────────────────────────────

ARROW_SCHEMA = pa.schema(
    [
        pa.field("customer_id", pa.string()),
        pa.field("transaction_amount", pa.float64()),
        pa.field("merchant_category", pa.string()),
        pa.field("device_type", pa.string()),
        pa.field("timestamp", pa.int64()),
        pa.field("date", pa.string()),  # partition column: YYYY-MM-DD
    ]
)


# ── Protobuf helpers (optional – graceful degradation) ────────────────────────

def _try_import_protobuf():
    """Return (Transaction proto class, enum maps) or None if not compiled."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "schemas" / "generated"))
        from transaction_pb2 import Transaction, MerchantCategory, DeviceType  # noqa: F401
        MERCHANT_MAP = {v: k.lower() for k, v in MerchantCategory.items() if v != 0}
        DEVICE_MAP = {v: k.lower() for k, v in DeviceType.items() if v != 0}
        return Transaction, MERCHANT_MAP, DEVICE_MAP
    except ImportError:
        return None, None, None


_PROTO_TRANSACTION, _MERCHANT_MAP, _DEVICE_MAP = _try_import_protobuf()


def parse_transaction(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a raw API response dict into a flat dict suitable for Arrow.
    Uses Protobuf validation when generated code is available.
    """
    if _PROTO_TRANSACTION is not None:
        msg = _PROTO_TRANSACTION()
        msg.customer_id = raw["customer_id"]
        msg.transaction_amount = float(raw["transaction_amount"])
        msg.timestamp = int(raw["timestamp"])
        # Validate & normalise category/device via proto enums
        cat_name = raw["merchant_category"].upper()
        dev_name = raw["device_type"].upper()
        mc_desc = _PROTO_TRANSACTION.DESCRIPTOR.fields_by_name["merchant_category"]
        dt_desc = _PROTO_TRANSACTION.DESCRIPTOR.fields_by_name["device_type"]
        mc_val = mc_desc.enum_type.values_by_name.get(cat_name)
        dt_val = dt_desc.enum_type.values_by_name.get(dev_name)
        merchant_category = mc_val.name.lower() if mc_val else raw["merchant_category"]
        device_type = dt_val.name.lower() if dt_val else raw["device_type"]
    else:
        merchant_category = raw["merchant_category"]
        device_type = raw["device_type"]

    ts = int(raw["timestamp"])
    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return {
        "customer_id": raw["customer_id"],
        "transaction_amount": float(raw["transaction_amount"]),
        "merchant_category": merchant_category,
        "device_type": device_type,
        "timestamp": ts,
        "date": date_str,
    }


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_batch(api_url: str, batch_size: int, retries: int = 3) -> List[Dict[str, Any]]:
    url = f"{api_url.rstrip('/')}/transactions?batch_size={batch_size}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retries, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch batch from {url} after {retries} attempts")


# ── Write helpers ─────────────────────────────────────────────────────────────

def write_parquet_partition(records: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Write a list of parsed transaction records as a Parquet file.
    Partitioned by ``date``, sorted by ``customer_id``.
    """
    if not records:
        return

    # Group by date partition
    from collections import defaultdict
    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)

    for date_str, rows in by_date.items():
        # Sort by customer_id within each partition
        rows.sort(key=lambda x: x["customer_id"])

        table = pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)

        partition_dir = output_dir / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"transactions_{int(time.time())}.parquet"
        out_path = partition_dir / file_name
        pq.write_table(table, out_path, compression="snappy")
        logger.info("Wrote %d rows to %s", len(rows), out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ingestion(
    api_url: str,
    batch_size: int,
    num_batches: int,
    output_dir: str,
    sleep_between_batches: float = 1.0,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for i in range(num_batches):
        logger.info("Fetching batch %d/%d (size=%d)…", i + 1, num_batches, batch_size)
        raw_records = fetch_batch(api_url, batch_size)
        parsed = [parse_transaction(r) for r in raw_records]
        write_parquet_partition(parsed, output_path)
        total += len(parsed)
        if i < num_batches - 1:
            time.sleep(sleep_between_batches)

    logger.info("Ingestion complete. Total records written: %d", total)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parquet ingestion pipeline")
    parser.add_argument("--api-url", default=os.getenv("DATA_API_URL", "http://localhost:8000"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--output-dir", default="data/parquet")
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ingestion(
        api_url=args.api_url,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        output_dir=args.output_dir,
        sleep_between_batches=args.sleep,
    )
