"""Packaging and deployment manifest helpers."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import PipelineConfig
from .utils import save_json, timestamp

LOGGER = logging.getLogger(__name__)


def create_deployment_manifest(model_path: str, config: PipelineConfig) -> Path:
    manifest = {
        "model_path": model_path,
        "model_version": config.project.get("version"),
        "serving": {
            "type": "batch_and_online",
            "latency_slo_ms": config.monitoring.get("latency_slo_ms"),
            "resources": {"cpu": 2, "memory_gb": 4},
        },
        "governance": {
            "approvers": [config.project.get("owner")],
            "created_at": timestamp(),
        },
    }
    path = Path("artifacts/deployment/manifest.json")
    save_json(manifest, path)
    LOGGER.info("Deployment manifest created at %s", path)
    return path


def export_runbook(config: PipelineConfig) -> None:
    runbook = Path("docs/RUNBOOK.md")
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text(
        f"""# Runbook: {config.project.get('name')}\n\n"
        "## Rollout Steps\n"
        "1. Deploy container image to staging.\n"
        "2. Run smoke tests and backfill batch jobs.\n"
        "3. Perform canary release to 10% traffic for 24h.\n"
        "4. Promote to production upon guardrail sign-off.\n\n"
        "## Rollback Plan\n"
        "- Trigger rollback job via CI pipeline.\n"
        "- Revert feature flags to previous model version.\n"
        "- Notify on-call ML engineer.\n\n"
        "## Monitoring & Alerts\n"
        f"- Latency SLO: {config.monitoring.get('latency_slo_ms')} ms.\n"
        f"- Drift threshold: {config.monitoring.get('drift_threshold')}.\n"
        f"- Alerts: {', '.join(config.monitoring.get('alert_emails', []))}.\n"
    )
    LOGGER.info("Runbook exported to %s", runbook)
