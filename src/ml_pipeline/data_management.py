"""Data ingestion, cataloging, and validation routines."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container with features, labels, and metadata."""

    features: pd.DataFrame
    labels: pd.Series
    metadata: Dict[str, str]


def ingest_dataset(config: PipelineConfig) -> DatasetBundle:
    """Load dataset according to config."""
    source = config.data.get("source")
    LOGGER.info("Ingesting dataset from %s", source)
    if source != "sklearn.datasets.load_breast_cancer":
        raise ValueError(f"Unsupported data source: {source}")

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.frame.drop(columns=[config.data.get("target", "target")])
    labels = dataset.target
    metadata = {
        "feature_names": ",".join(dataset.feature_names),
        "target_names": ",".join(dataset.target_names),
        "n_samples": str(len(dataset.frame)),
        "n_features": str(len(dataset.feature_names)),
    }
    return DatasetBundle(features=features, labels=labels, metadata=metadata)


def validate_dataset(bundle: DatasetBundle, config: PipelineConfig) -> None:
    """Run a suite of validation checks against the dataset."""
    LOGGER.info("Validating dataset against schema and quality gates")
    expected_schema = config.data.get("schema", {})
    missing_columns = set(expected_schema) - set(bundle.features.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if bundle.features.isnull().mean().max() > config.data["validations"].get(
        "max_missing_ratio", 0
    ):
        raise ValueError("Missing data ratio exceeds threshold")

    if not config.data["validations"].get("allow_negative", True):
        if (bundle.features < 0).any().any():
            raise ValueError("Negative values found but not allowed")

    min_rows = config.data["validations"].get("min_rows")
    if min_rows and len(bundle.features) < min_rows:
        raise ValueError(
            f"Dataset row count {len(bundle.features)} below minimum {min_rows}"
        )

    LOGGER.info("Dataset validation passed")


def train_validation_test_split(
    bundle: DatasetBundle, config: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset into train/validation/test sets."""
    from sklearn.model_selection import train_test_split

    LOGGER.info("Creating train/validation/test splits")
    stratify = bundle.labels if config.splits.get("stratify", False) else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        bundle.features,
        bundle.labels,
        test_size=config.splits.get("test_size", 0.2),
        random_state=config.splits.get("random_state", 42),
        stratify=stratify,
    )

    stratify_val = y_temp if config.splits.get("stratify", False) else None
    validation_size = config.splits.get("validation_size", 0.2)
    relative_val_size = validation_size / (1 - config.splits.get("test_size", 0.2))

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=config.splits.get("random_state", 42),
        stratify=stratify_val,
    )

    LOGGER.info(
        "Split sizes -> train: %s, validation: %s, test: %s",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
