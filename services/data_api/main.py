"""
FastAPI service that generates synthetic transaction data and fraud labels.
"""

import random
import time
from typing import List, Optional

from faker import Faker
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

fake = Faker()

app = FastAPI(title="Fraud Detection Data API", version="1.0.0")

# ── Constants ────────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = ["electronics", "grocery", "travel", "dining", "entertainment", "fashion", "health"]
DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos_terminal"]

# Seed a small pool of customer IDs so labels can be looked up deterministically
CUSTOMER_POOL = [str(fake.unique.random_int(min=1000, max=9999)) for _ in range(200)]
fake.unique.clear()

# Pre-assign ~7 % of customers as fraudsters (stable across requests)
random.seed(42)
FRAUD_CUSTOMERS = set(random.sample(CUSTOMER_POOL, k=int(len(CUSTOMER_POOL) * 0.07)))
random.seed()  # re-seed with system entropy


# ── Pydantic models ───────────────────────────────────────────────────────────

class Transaction(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    transaction_amount: float = Field(..., ge=0.01, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="Merchant category")
    device_type: str = Field(..., description="Device used for the transaction")
    timestamp: int = Field(..., description="Unix timestamp of the transaction")


class Label(BaseModel):
    customer_id: str
    label: str = Field(..., description="'fraud' or 'non_fraud'")


class HealthResponse(BaseModel):
    status: str
    timestamp: int


# ── Helper ────────────────────────────────────────────────────────────────────

def _generate_transaction(customer_id: Optional[str] = None) -> Transaction:
    cid = customer_id or random.choice(CUSTOMER_POOL)
    # Fraudulent transactions tend to be higher-value
    is_fraud = cid in FRAUD_CUSTOMERS
    amount = (
        round(random.uniform(500.0, 5000.0), 2) if is_fraud
        else round(random.uniform(1.0, 1000.0), 2)
    )
    return Transaction(
        customer_id=cid,
        transaction_amount=amount,
        merchant_category=random.choice(MERCHANT_CATEGORIES),
        device_type=random.choice(DEVICE_TYPES),
        timestamp=int(time.time()) - random.randint(0, 3600),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness probe."""
    return HealthResponse(status="ok", timestamp=int(time.time()))


@app.get("/transaction", response_model=Transaction, tags=["data"])
def get_transaction(customer_id: Optional[str] = Query(default=None)):
    """Return a single synthetic transaction."""
    return _generate_transaction(customer_id)


@app.get("/transactions", response_model=List[Transaction], tags=["data"])
def get_transactions(batch_size: int = Query(default=10, ge=1, le=500)):
    """Return a batch of synthetic transactions."""
    return [_generate_transaction() for _ in range(batch_size)]


@app.get("/labels", response_model=List[Label], tags=["data"])
def get_labels(customer_ids: Optional[str] = Query(default=None, description="Comma-separated customer IDs")):
    """
    Return fraud / non_fraud labels for a list of customer IDs.
    If no IDs are supplied, labels for the full customer pool are returned.
    """
    if customer_ids:
        ids = [cid.strip() for cid in customer_ids.split(",") if cid.strip()]
    else:
        ids = CUSTOMER_POOL
    return [
        Label(customer_id=cid, label="fraud" if cid in FRAUD_CUSTOMERS else "non_fraud")
        for cid in ids
    ]


@app.get("/label/{customer_id}", response_model=Label, tags=["data"])
def get_label(customer_id: str):
    """Return the fraud label for a single customer ID."""
    return Label(
        customer_id=customer_id,
        label="fraud" if customer_id in FRAUD_CUSTOMERS else "non_fraud",
    )
