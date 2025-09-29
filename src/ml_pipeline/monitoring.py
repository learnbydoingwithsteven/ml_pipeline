"""Monitoring utilities to simulate drift and alerting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .utils import timestamp

LOGGER = logging.getLogger(__name__)


def simulate_monitoring_signals(
    baseline_predictions: np.ndarray,
    new_predictions: np.ndarray,
    config_guardrails: Dict[str, float],
) -> Dict[str, float]:
    """Compute a simple drift score between two prediction windows."""
    drift = float(abs(baseline_predictions.mean() - new_predictions.mean()))
    status = "OK" if drift < config_guardrails.get("drift_threshold", 0.1) else "ALERT"
    metrics = {
        "prediction_drift": drift,
        "status": status,
        "generated_at": timestamp(),
    }
    return metrics


def write_monitoring_dashboard(metrics: Dict[str, float]) -> None:
    dashboard = Path("docs/MONITORING_DASHBOARD.md")
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.write_text(
        "# Monitoring Dashboard\n\n"
        f"- Generated: {metrics['generated_at']}\n"
        f"- Prediction drift: {metrics['prediction_drift']:.4f}\n"
        f"- Status: {metrics['status']}\n"
    )
    LOGGER.info("Monitoring dashboard saved to %s", dashboard)
