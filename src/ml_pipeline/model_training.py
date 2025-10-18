"""Model training and selection utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    estimator: LogisticRegression
    metrics: Dict[str, float]
    path: str


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: PipelineConfig,
) -> TrainedModel:
    """Train model with hyperparameter search and evaluate on validation set."""
    LOGGER.info("Training model with hyperparameter search")
    hyperparameters = config.training.get("hyperparameters", {"C": 1.0})
    param_grid = {"C": [hyperparameters.get("C", 1.0), hyperparameters.get("C", 1.0) * 0.5, hyperparameters.get("C", 1.0) * 2]}

    base_model = LogisticRegression(
        penalty=hyperparameters.get("penalty", "l2"),
        solver=hyperparameters.get("solver", "lbfgs"),
        max_iter=hyperparameters.get("max_iter", 1000),
        class_weight=hyperparameters.get("class_weight"),
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=config.experiment.get("primary_metric", "roc_auc"),
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    LOGGER.info("Best hyperparameters: %s", search.best_params_)

    best_model: LogisticRegression = search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred),
        "recall": recall_score(y_val, y_val_pred),
        "roc_auc": roc_auc_score(y_val, y_val_proba),
    }
    LOGGER.info("Validation metrics: %s", metrics)
    _enforce_quality_gates(metrics, config)

    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    LOGGER.info("Saved trained model to %s", model_path)

    return TrainedModel(estimator=best_model, metrics=metrics, path=str(model_path))


def _enforce_quality_gates(metrics: Dict[str, float], config: PipelineConfig) -> None:
    LOGGER.info("Checking quality gates against guardrail metrics")
    for guardrail in config.experiment.get("guardrails", []):
        metric_name = guardrail["metric"]
        threshold = guardrail["threshold"]
        if metrics.get(metric_name, 0.0) < threshold:
            raise ValueError(
                f"Quality gate failed: {metric_name}={metrics.get(metric_name):.3f} < {threshold}"
            )
    LOGGER.info("All quality gates passed")
