"""High level orchestration of the ML pipeline."""
from __future__ import annotations

import logging
import numpy as np

from .config import PipelineConfig, load_config
from .data_management import (
    DatasetBundle,
    ingest_dataset,
    train_validation_test_split,
    validate_dataset,
)
from .evaluation import evaluate_on_test_set
from .feature_engineering import build_feature_pipeline, transform_features
from .model_training import train_model
from .monitoring import simulate_monitoring_signals, write_monitoring_dashboard
from .packaging import create_deployment_manifest, export_runbook
from .utils import setup_logging

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/pipeline.yaml") -> None:
    setup_logging()
    config = load_config(config_path)

    LOGGER.info("Starting pipeline for %s", config.project.get("name"))

    # Stage 1: Data ingestion & validation
    bundle = ingest_dataset(config)
    validate_dataset(bundle, config)

    # Stage 2: Splitting
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
        bundle, config
    )

    # Stage 3: Feature engineering
    feature_pipeline, X_train_transformed = build_feature_pipeline(X_train, config)
    X_val_transformed = transform_features(feature_pipeline, X_val)
    X_test_transformed = transform_features(feature_pipeline, X_test)

    # Stage 4: Model training and selection
    trained_model = train_model(
        X_train_transformed, y_train, X_val_transformed, y_val, config
    )

    # Stage 5: Evaluation
    evaluate_on_test_set(trained_model.estimator, X_test_transformed, y_test, config)

    # Stage 6: Packaging and documentation
    create_deployment_manifest(trained_model.path, config)
    export_runbook(config)

    # Stage 7: Monitoring simulation
    baseline_preds = trained_model.estimator.predict_proba(X_val_transformed)[:, 1]
    new_preds = trained_model.estimator.predict_proba(X_test_transformed)[:, 1]
    monitoring_metrics = simulate_monitoring_signals(
        baseline_preds,
        new_preds,
        config.monitoring,
    )
    write_monitoring_dashboard(monitoring_metrics)

    LOGGER.info("Pipeline run complete")


if __name__ == "__main__":
    run_pipeline()
