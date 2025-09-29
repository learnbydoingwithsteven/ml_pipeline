"""Streamlit UI for interacting with the ML pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ml_pipeline.config import load_config
from src.ml_pipeline.pipeline import run_pipeline

CONFIG_PATH = Path("configs/pipeline.yaml")


def load_metrics() -> dict:
    metrics_path = Path("artifacts/evaluation/metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def load_classification_report() -> pd.DataFrame:
    report_path = Path("artifacts/evaluation/classification_report.json")
    if report_path.exists():
        raw = json.loads(report_path.read_text())
        return pd.DataFrame(raw).T
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="ML Pipeline Control Center", layout="wide")
    config = load_config(CONFIG_PATH)

    st.title("ML Pipeline Control Center")
    st.caption(config.project.get("description"))

    with st.sidebar:
        st.header("Pipeline Controls")
        if st.button("Run pipeline"):
            with st.spinner("Executing pipeline stages..."):
                run_pipeline(str(CONFIG_PATH))
            st.success("Pipeline execution completed")

        st.subheader("Project metadata")
        st.json(config.project)

    metrics = load_metrics()
    if metrics:
        st.subheader("Latest Evaluation Metrics")
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    else:
        st.info("Run the pipeline to populate evaluation metrics.")

    report_df = load_classification_report()
    if not report_df.empty:
        st.subheader("Classification Report")
        st.dataframe(report_df.style.format("{:.3f}"))

        if {"precision", "recall"}.issubset(report_df.columns):
            chart = px.bar(
                report_df.reset_index(),
                x="index",
                y=["precision", "recall", "f1-score"],
                barmode="group",
                title="Precision/Recall/F1 by class",
            )
            st.plotly_chart(chart, use_container_width=True)

    model_card = Path("docs/MODEL_CARD.md")
    if model_card.exists():
        st.subheader("Model Card")
        st.markdown(model_card.read_text())

    monitoring_dashboard = Path("docs/MONITORING_DASHBOARD.md")
    if monitoring_dashboard.exists():
        st.subheader("Monitoring Dashboard")
        st.markdown(monitoring_dashboard.read_text())


if __name__ == "__main__":
    main()
