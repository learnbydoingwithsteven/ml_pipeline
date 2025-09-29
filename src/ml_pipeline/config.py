"""Configuration utilities for the ML pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    """Dataclass wrapper around the YAML configuration."""

    raw: Dict[str, Any]

    @property
    def project(self) -> Dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def splits(self) -> Dict[str, Any]:
        return self.raw.get("splits", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def experiment(self) -> Dict[str, Any]:
        return self.raw.get("experiment", {})

    @property
    def monitoring(self) -> Dict[str, Any]:
        return self.raw.get("monitoring", {})

    @property
    def ui(self) -> Dict[str, Any]:
        return self.raw.get("ui", {})


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return PipelineConfig(raw=data)
