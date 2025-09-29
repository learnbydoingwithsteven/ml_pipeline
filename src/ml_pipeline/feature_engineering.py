"""Feature engineering routines."""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def build_feature_pipeline(
    X_train: pd.DataFrame, config: PipelineConfig
) -> Tuple[Pipeline, pd.DataFrame]:
    """Create feature engineering pipeline and fit it."""
    LOGGER.info("Building feature engineering pipeline")

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "selector",
                            VarianceThreshold(
                                threshold=config.features["feature_selection"].get(
                                    "threshold", 0.0
                                )
                            ),
                        ),
                    ]
                ),
                numeric_columns,
            )
        ]
    )

    pipeline = Pipeline([("features", transformer)])
    pipeline.fit(X_train)
    transformed = pipeline.transform(X_train)
    transformed_df = pd.DataFrame(
        transformed,
        columns=[f"f_{idx}" for idx in range(transformed.shape[1])],
        index=X_train.index,
    )
    LOGGER.info("Feature pipeline fitted with %s features", transformed_df.shape[1])
    return pipeline, transformed_df


def transform_features(
    pipeline: Pipeline, X: pd.DataFrame
) -> pd.DataFrame:
    """Apply feature pipeline to dataset."""
    transformed = pipeline.transform(X)
    return pd.DataFrame(
        transformed,
        columns=[f"f_{idx}" for idx in range(transformed.shape[1])],
        index=X.index,
    )
