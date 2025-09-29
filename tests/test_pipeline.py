"""Smoke tests for the ML pipeline."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

np = pytest.importorskip('numpy')
sklearn = pytest.importorskip('sklearn')

from src.ml_pipeline.pipeline import run_pipeline


def test_pipeline_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the pipeline end-to-end and ensure artifacts are created."""
    # Copy config to temporary directory to avoid contaminating repo artifacts
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(Path("configs/pipeline.yaml").read_text())

    # Change working directory to tmp for artifact isolation
    monkeypatch.chdir(tmp_path)

    run_pipeline(str(config_path))

    assert (tmp_path / "artifacts/models/best_model.joblib").exists()
    assert (tmp_path / "docs/MODEL_CARD.md").exists()
    assert (tmp_path / "docs/MONITORING_DASHBOARD.md").exists()
