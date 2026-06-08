"""
LightGBM fraud classifier with hyperparameter tuning.

Pipeline
--------
1. Load features from gold.ml_features Iceberg table (or Parquet fallback).
2. Build an sklearn Pipeline:
      OrdinalEncoder  →  LGBMClassifier
3. Tune hyperparameters with Optuna, maximising average_precision_score
   (area under the precision-recall curve).
4. Log every trial to MLflow; register the best model as "fraud-classifier".

Usage
-----
    python training/train.py \
        --data-path data/parquet \
        --mlflow-uri http://localhost:5000 \
        --n-trials 30
"""

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Feature catalogue ─────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["merchant_category", "device_type"]
NUMERIC_COLS = [
    "transaction_amount",
    "hour_of_day",
    "day_of_week",
    "txn_count_1h",
    "total_amount_1h",
    "txn_count_24h",
    "total_amount_24h",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "label"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_iceberg(catalog_uri: str) -> pd.DataFrame:
    """Load gold features via PyIceberg."""
    from pyiceberg.catalog import load_catalog
    catalog = load_catalog(
        "default",
        **{"uri": catalog_uri,
           "s3.endpoint": os.getenv("S3_ENDPOINT", "http://localhost:9000"),
           "s3.access-key-id": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
           "s3.secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")},
    )
    tbl = catalog.load_table("gold.ml_features")
    return tbl.scan().to_pandas()


def load_from_parquet(data_path: str) -> pd.DataFrame:
    """Load features from Parquet files (fallback)."""
    import pyarrow.parquet as pq
    dataset = pq.read_table(data_path)
    return dataset.to_pandas()


def load_data(data_path: Optional[str] = None, catalog_uri: Optional[str] = None) -> pd.DataFrame:
    if catalog_uri:
        try:
            logger.info("Loading features from Iceberg catalog…")
            return load_from_iceberg(catalog_uri)
        except Exception as exc:
            logger.warning("Iceberg load failed (%s). Falling back to Parquet.", exc)
    if data_path:
        logger.info("Loading features from Parquet at %s…", data_path)
        return load_from_parquet(data_path)
    raise ValueError("Either --data-path or --catalog-uri must be provided.")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Ensure required columns exist (fill missing with defaults for robustness)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "unknown"

    X = df[FEATURE_COLS].copy()
    y = (df[TARGET_COL].str.lower() == "fraud").astype(int)
    return X, y


def build_pipeline(params: Dict[str, Any], scale_pos_weight: float) -> Pipeline:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        categories="auto",
    )
    clf = LGBMClassifier(
        objective="binary",
        metric="average_precision",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        **params,
    )
    return Pipeline(steps=[
        ("encoder", encoder),
        ("classifier", clf),
    ])


# ── Optuna objective ──────────────────────────────────────────────────────────

def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
    cv_folds: int,
) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    pipe = build_pipeline(params, scale_pos_weight)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1
    )
    return float(scores.mean())


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    mlflow_uri: str,
    n_trials: int = 30,
    cv_folds: int = 5,
    experiment_name: str = "fraud-detection",
) -> Pipeline:
    X, y = prepare_xy(df)

    n_positive = int(y.sum())
    n_negative = int((y == 0).sum())
    scale_pos_weight = n_negative / max(n_positive, 1)
    logger.info("Class balance: pos=%d neg=%d  scale_pos_weight=%.3f", n_positive, n_negative, scale_pos_weight)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    study = optuna.create_study(direction="maximize", study_name="lgbm-ap")

    def _logged_objective(trial: optuna.Trial) -> float:
        ap = _objective(trial, X, y, scale_pos_weight, cv_folds)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_average_precision", ap)
        return ap

    with mlflow.start_run(run_name="lgbm-hyperparameter-search"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_positive", n_positive)
        mlflow.log_param("n_negative", n_negative)

        study.optimize(_logged_objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info("Best params: %s", best_params)
        logger.info("Best CV average_precision: %.4f", study.best_value)

        # Retrain on full dataset with best params
        best_pipe = build_pipeline(best_params, scale_pos_weight)
        best_pipe.fit(X, y)

        # Compute final in-sample metrics
        y_pred = best_pipe.predict(X)
        y_prob = best_pipe.predict_proba(X)[:, 1]
        ap = average_precision_score(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_pred, zero_division=0)

        mlflow.log_params(best_params)
        mlflow.log_metric("train_average_precision", ap)
        mlflow.log_metric("train_roc_auc", auc)
        mlflow.log_metric("train_f1", f1)
        mlflow.log_metric("best_cv_average_precision", study.best_value)

        # Log the full sklearn pipeline
        signature = mlflow.models.infer_signature(X, y_prob)
        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            signature=signature,
            registered_model_name="fraud-classifier",
        )
        logger.info("Model logged: %s", model_info.model_uri)

        # Transition the latest version to Production
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("fraud-classifier", stages=["None"])
        if versions:
            client.transition_model_version_stage(
                name="fraud-classifier",
                version=versions[0].version,
                stage="Production",
            )
            logger.info("Transitioned version %s to Production.", versions[0].version)

        # Persist model locally as well
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        import joblib
        joblib.dump(best_pipe, model_dir / "fraud_classifier.pkl")
        logger.info("Model saved to %s", model_dir / "fraud_classifier.pkl")

    return best_pipe


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM fraud classifier")
    parser.add_argument("--data-path", default=os.getenv("FEATURE_DATA_PATH", "data/parquet"))
    parser.add_argument("--catalog-uri", default=os.getenv("ICEBERG_CATALOG_URI", ""))
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--experiment", default="fraud-detection")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = load_data(
        data_path=args.data_path if args.data_path else None,
        catalog_uri=args.catalog_uri if args.catalog_uri else None,
    )
    train(
        df=df,
        mlflow_uri=args.mlflow_uri,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        experiment_name=args.experiment,
    )
