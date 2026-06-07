"""
Model Registry utilities – thin wrappers around the MLflow client.

Provides helpers to:
- retrieve the Production model pipeline
- list all versions of a registered model
- compare runs by average_precision and promote the best one
- archive stale Production versions

Usage (as a script)
-------------------
    python registry/model_registry.py --action promote \
        --model-name fraud-classifier \
        --mlflow-uri http://localhost:5000
"""

import argparse
import logging
import os
from typing import List, Optional

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fraud-classifier"
METRIC_KEY = "train_average_precision"


class ModelRegistry:
    """High-level wrapper around the MLflow Model Registry."""

    def __init__(self, tracking_uri: str, model_name: str = DEFAULT_MODEL):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name

    # ── Read operations ───────────────────────────────────────────────────────

    def list_versions(self, stage: Optional[str] = None) -> List:
        """Return all (or stage-filtered) versions of the registered model."""
        if stage:
            return self.client.get_latest_versions(self.model_name, stages=[stage])
        return self.client.search_model_versions(f"name='{self.model_name}'")

    def get_production_uri(self) -> Optional[str]:
        """Return the model URI for the current Production version."""
        versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if not versions:
            logger.warning("No Production version found for '%s'.", self.model_name)
            return None
        v = versions[0]
        uri = f"models:/{self.model_name}/{v.version}"
        logger.info("Production model URI: %s", uri)
        return uri

    def load_production_model(self):
        """Load and return the Production sklearn pipeline."""
        uri = self.get_production_uri()
        if uri is None:
            raise RuntimeError(f"No Production model registered for '{self.model_name}'")
        logger.info("Loading model from %s…", uri)
        return mlflow.sklearn.load_model(uri)

    # ── Promotion operations ──────────────────────────────────────────────────

    def promote_best_by_metric(self, experiment_name: str = "fraud-detection") -> None:
        """
        Find the run with the highest ``train_average_precision`` in the given
        experiment and promote its registered model version to Production.
        All other Production versions are archived.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.{METRIC_KEY} > 0",
            order_by=[f"metrics.{METRIC_KEY} DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("No eligible runs found in experiment '%s'.", experiment_name)
            return

        best_run = runs[0]
        best_ap = best_run.data.metrics.get(METRIC_KEY, 0.0)
        best_run_id = best_run.info.run_id
        logger.info("Best run: %s  %s=%.4f", best_run_id, METRIC_KEY, best_ap)

        # Find the model version linked to this run
        all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        target_version = None
        for v in all_versions:
            if v.run_id == best_run_id:
                target_version = v
                break

        if target_version is None:
            logger.error("No registered version found for run %s.", best_run_id)
            return

        # Archive existing Production versions
        current_prod = self.client.get_latest_versions(self.model_name, stages=["Production"])
        for v in current_prod:
            if v.version != target_version.version:
                logger.info("Archiving version %s.", v.version)
                self.client.transition_model_version_stage(
                    name=self.model_name, version=v.version, stage="Archived"
                )

        # Promote best version
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=target_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Version %s promoted to Production (AP=%.4f).",
            target_version.version, best_ap,
        )

    def set_model_tag(self, version: str, key: str, value: str) -> None:
        self.client.set_model_version_tag(self.model_name, version, key, value)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--action", choices=["list", "promote", "get-uri"], default="list")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--experiment", default="fraud-detection")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    registry = ModelRegistry(args.mlflow_uri, args.model_name)

    if args.action == "list":
        for v in registry.list_versions():
            print(f"version={v.version}  stage={v.current_stage}  run={v.run_id}")
    elif args.action == "promote":
        registry.promote_best_by_metric(args.experiment)
    elif args.action == "get-uri":
        print(registry.get_production_uri())
