"""
Schema Registry utilities – manages Protobuf schemas in Confluent Schema Registry.

Features
--------
- Register the transaction Protobuf schema under ``transactions-value``
- Enforce BACKWARD compatibility (new optional fields won't break consumers)
- Retrieve the latest schema version
- Check compatibility before publishing a new schema version

Usage (as a script)
-------------------
    python registry/schema_registry.py --action register \
        --registry-url http://localhost:8085 \
        --proto-path schemas/transaction.proto

    python registry/schema_registry.py --action check-compat \
        --registry-url http://localhost:8085 \
        --proto-path schemas/transaction.proto
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SUBJECT = "transactions-value"
COMPATIBILITY_MODE = "BACKWARD"


class SchemaRegistryClient:
    """Minimal HTTP client for Confluent Schema Registry REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/vnd.schemaregistry.v1+json"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _handle(self, resp: requests.Response) -> Dict[str, Any]:
        if not resp.ok:
            raise RuntimeError(f"Schema Registry error {resp.status_code}: {resp.text}")
        return resp.json()

    # ── Schema operations ─────────────────────────────────────────────────────

    def get_latest_schema(self, subject: str) -> Optional[Dict[str, Any]]:
        resp = self.session.get(self._url(f"subjects/{subject}/versions/latest"))
        if resp.status_code == 404:
            return None
        return self._handle(resp)

    def register_schema(self, subject: str, schema_str: str, schema_type: str = "PROTOBUF") -> int:
        payload = {"schemaType": schema_type, "schema": schema_str}
        resp = self.session.post(self._url(f"subjects/{subject}/versions"), json=payload)
        result = self._handle(resp)
        schema_id = result["id"]
        logger.info("Registered schema for subject '%s' with ID %d.", subject, schema_id)
        return schema_id

    def check_compatibility(
        self, subject: str, schema_str: str, schema_type: str = "PROTOBUF"
    ) -> bool:
        """Return True if the new schema is backward-compatible with the latest registered version."""
        payload = {"schemaType": schema_type, "schema": schema_str}
        resp = self.session.post(
            self._url(f"compatibility/subjects/{subject}/versions/latest"), json=payload
        )
        if resp.status_code == 404:
            logger.info("No existing schema for '%s'; new schema is trivially compatible.", subject)
            return True
        result = self._handle(resp)
        is_compatible = result.get("is_compatible", False)
        logger.info(
            "Compatibility check for '%s': %s",
            subject,
            "COMPATIBLE" if is_compatible else "INCOMPATIBLE",
        )
        return is_compatible

    # ── Compatibility config ──────────────────────────────────────────────────

    def set_compatibility(self, subject: str, mode: str = COMPATIBILITY_MODE) -> None:
        payload = {"compatibility": mode}
        resp = self.session.put(self._url(f"config/{subject}"), json=payload)
        self._handle(resp)
        logger.info("Set compatibility for '%s' to %s.", subject, mode)

    def get_compatibility(self, subject: str) -> str:
        resp = self.session.get(self._url(f"config/{subject}"))
        if resp.status_code == 404:
            # Fall back to global config
            resp = self.session.get(self._url("config"))
        result = self._handle(resp)
        return result.get("compatibilityLevel", result.get("compatibility", "UNKNOWN"))


# ── High-level operations ─────────────────────────────────────────────────────

def register_transaction_schema(
    registry_url: str,
    proto_path: str,
    subject: str = DEFAULT_SUBJECT,
) -> int:
    client = SchemaRegistryClient(registry_url)
    schema_str = Path(proto_path).read_text()

    # Ensure BACKWARD compatibility is set before registering
    client.set_compatibility(subject, COMPATIBILITY_MODE)

    # Check compatibility first
    if not client.check_compatibility(subject, schema_str):
        raise RuntimeError(
            f"Schema at '{proto_path}' is NOT backward-compatible with the "
            f"latest version registered under '{subject}'. Aborting."
        )

    schema_id = client.register_schema(subject, schema_str)
    return schema_id


def show_latest_schema(registry_url: str, subject: str = DEFAULT_SUBJECT) -> None:
    client = SchemaRegistryClient(registry_url)
    schema = client.get_latest_schema(subject)
    if schema:
        import json
        print(json.dumps(schema, indent=2))
    else:
        logger.info("No schema registered for subject '%s'.", subject)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Schema Registry CLI")
    parser.add_argument(
        "--action",
        choices=["register", "check-compat", "get-latest", "set-compat"],
        default="register",
    )
    parser.add_argument("--registry-url", default=os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8085"))
    parser.add_argument("--proto-path", default="schemas/transaction.proto")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--compat-mode", default=COMPATIBILITY_MODE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.action == "register":
        sid = register_transaction_schema(args.registry_url, args.proto_path, args.subject)
        print(f"Schema ID: {sid}")
    elif args.action == "check-compat":
        client = SchemaRegistryClient(args.registry_url)
        schema_str = Path(args.proto_path).read_text()
        result = client.check_compatibility(args.subject, schema_str)
        print("compatible" if result else "incompatible")
    elif args.action == "get-latest":
        show_latest_schema(args.registry_url, args.subject)
    elif args.action == "set-compat":
        client = SchemaRegistryClient(args.registry_url)
        client.set_compatibility(args.subject, args.compat_mode)
