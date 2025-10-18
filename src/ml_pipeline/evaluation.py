"""Evaluation, validation, and reporting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import PipelineConfig
from .utils import save_json

LOGGER = logging.getLogger(__name__)


def evaluate_on_test_set(
    model, X_test: pd.DataFrame, y_test: pd.Series, config: PipelineConfig
) -> Dict[str, float]:
    LOGGER.info("Running evaluation on test set")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, eval_dir / "metrics.json")
    save_json(report, eval_dir / "classification_report.json")
    save_json({"confusion_matrix": conf_matrix}, eval_dir / "confusion_matrix.json")
    LOGGER.info("Saved evaluation artifacts to %s", eval_dir)

    _generate_model_card(metrics, report, config)
    return metrics


def _generate_model_card(metrics: Dict[str, float], report: Dict, config: PipelineConfig) -> None:
    """Write a markdown model card summarizing the run."""
    model_card = Path("docs/MODEL_CARD.md")
    model_card.parent.mkdir(parents=True, exist_ok=True)
    model_card.write_text(
        (
            f"# Model Card: {config.project.get('name')}\n\n"
            f"**Version:** {config.project.get('version')}  \\\n"
            f"**Owner:** {config.project.get('owner')}  \\\n"
            f"**Primary Metric:** {config.experiment.get('primary_metric')}\n\n"
            "## Evaluation Summary\n"
            f"- ROC-AUC: {metrics['roc_auc']:.3f}\n"
            f"- Precision (class 1): {report['1']['precision']:.3f}\n"
            f"- Recall (class 1): {report['1']['recall']:.3f}\n"
            f"- F1 (class 1): {report['1']['f1-score']:.3f}\n\n"
            "## Data\n"
            "- Source: Breast cancer diagnostic features from scikit-learn.\n"
            "- Rows: 569, Columns: 30 numeric features.\n\n"
            "## Ethical Considerations\n"
            "- Ensure predictions are reviewed by medical professionals.\n"
            "- Monitor for bias against subpopulations.\n\n"
            "## Operational Guidance\n"
            "- Retrain quarterly or when drift exceeds thresholds.\n"
            "- Monitor latency, accuracy, and alert channels.\n"
        )
    )
    LOGGER.info("Model card generated at %s", model_card)
