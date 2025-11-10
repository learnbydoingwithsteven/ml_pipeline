"""Streamlit UI for interacting with the ML pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def load_confusion_matrix() -> dict:
    cm_path = Path("artifacts/evaluation/confusion_matrix.json")
    if cm_path.exists():
        return json.loads(cm_path.read_text())
    return {}


def check_artifacts_exist() -> dict:
    """Check which pipeline artifacts exist."""
    return {
        "data_validated": True,  # Assume if we're running
        "model_trained": Path("artifacts/models/best_model.joblib").exists(),
        "evaluation_done": Path("artifacts/evaluation/metrics.json").exists(),
        "deployment_ready": Path("artifacts/deployment/manifest.json").exists(),
        "monitoring_setup": Path("docs/MONITORING_DASHBOARD.md").exists(),
    }


def render_pipeline_overview():
    """Render the pipeline stages overview."""
    st.header("🔄 ML Pipeline Workflow")
    st.markdown("""
    This pipeline demonstrates a **production-grade machine learning workflow** with governance, 
    quality gates, and monitoring. Each stage is designed following MLOps best practices.
    """)
    
    # Pipeline stages with icons and descriptions
    stages = [
        {
            "icon": "📥",
            "name": "Data Ingestion & Validation",
            "description": "Load data from source and validate against schema, quality checks, and business rules",
            "key_points": ["Schema validation", "Data quality gates", "Freshness checks"]
        },
        {
            "icon": "✂️",
            "name": "Data Splitting",
            "description": "Split data into train, validation, and test sets with stratification for balanced classes",
            "key_points": ["Stratified splits", "Reproducible (seed=42)", "60/20/20 split"]
        },
        {
            "icon": "🔧",
            "name": "Feature Engineering",
            "description": "Transform raw features through scaling and selection to improve model performance",
            "key_points": ["StandardScaler normalization", "Variance threshold selection", "Pipeline architecture"]
        },
        {
            "icon": "🎯",
            "name": "Model Training",
            "description": "Train model with hyperparameter tuning and validate against quality guardrails",
            "key_points": ["Hyperparameter search", "Cross-validation", "Quality gate enforcement"]
        },
        {
            "icon": "📊",
            "name": "Evaluation",
            "description": "Comprehensive testing on held-out data with multiple metrics and visualizations",
            "key_points": ["ROC-AUC, Precision, Recall", "Confusion matrix", "Model card generation"]
        },
        {
            "icon": "📦",
            "name": "Packaging",
            "description": "Create deployment artifacts, manifests, and operational documentation",
            "key_points": ["Deployment manifest", "Runbook creation", "Model versioning"]
        },
        {
            "icon": "📈",
            "name": "Monitoring",
            "description": "Set up monitoring for drift detection, performance tracking, and alerting",
            "key_points": ["Drift detection", "Performance SLOs", "Alert configuration"]
        }
    ]
    
    # Display stages in expandable sections
    for i, stage in enumerate(stages, 1):
        with st.expander(f"{stage['icon']} **Stage {i}: {stage['name']}**", expanded=False):
            st.markdown(f"**Description:** {stage['description']}")
            st.markdown("**Key Components:**")
            for point in stage['key_points']:
                st.markdown(f"- {point}")


def render_project_info(config):
    """Render project information and configuration."""
    st.header("📋 Project Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Project Name", config.project.get("name", "N/A"))
        st.metric("Version", config.project.get("version", "N/A"))
    
    with col2:
        st.metric("Owner", config.project.get("owner", "N/A"))
        st.metric("Algorithm", config.training.get("algorithm", "N/A").replace("_", " ").title())
    
    with col3:
        st.metric("Primary Metric", config.experiment.get("primary_metric", "N/A").upper())
        st.metric("Data Source", "Sklearn Breast Cancer" if "sklearn" in config.data.get("source", "") else "Custom")
    
    st.markdown("---")
    st.markdown(f"**Description:** {config.project.get('description', '')}")


def render_data_insights(config):
    """Render data configuration and validation rules."""
    st.header("📊 Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Schema")
        st.markdown(f"**Features:** {len(config.data.get('schema', {}))} numeric features")
        
        # Show feature categories
        features = list(config.data.get('schema', {}).keys())
        mean_features = [f for f in features if f.startswith('mean')]
        error_features = [f for f in features if 'error' in f]
        worst_features = [f for f in features if f.startswith('worst')]
        
        st.markdown(f"- **Mean features:** {len(mean_features)}")
        st.markdown(f"- **Error features:** {len(error_features)}")
        st.markdown(f"- **Worst features:** {len(worst_features)}")
    
    with col2:
        st.subheader("Validation Rules")
        validations = config.data.get('validations', {})
        st.markdown(f"- **Minimum rows:** {validations.get('min_rows', 'N/A')}")
        st.markdown(f"- **Max missing ratio:** {validations.get('max_missing_ratio', 'N/A')}")
        st.markdown(f"- **Allow negative:** {validations.get('allow_negative', 'N/A')}")
        
        st.subheader("Split Configuration")
        splits = config.splits
        st.markdown(f"- **Test size:** {splits.get('test_size', 0.2)*100:.0f}%")
        st.markdown(f"- **Validation size:** {splits.get('validation_size', 0.2)*100:.0f}%")
        st.markdown(f"- **Stratified:** {'✅' if splits.get('stratify') else '❌'}")


def render_training_config(config):
    """Render model training configuration."""
    st.header("🎯 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameters")
        hyperparams = config.training.get('hyperparameters', {})
        for key, value in hyperparams.items():
            st.markdown(f"- **{key}:** `{value}`")
    
    with col2:
        st.subheader("Quality Guardrails")
        guardrails = config.experiment.get('guardrails', [])
        st.markdown(f"**Primary Metric:** {config.experiment.get('primary_metric', 'N/A').upper()}")
        st.markdown("**Minimum Thresholds:**")
        for guardrail in guardrails:
            metric = guardrail.get('metric', 'N/A')
            threshold = guardrail.get('threshold', 0)
            st.markdown(f"- **{metric.capitalize()}:** ≥ {threshold:.1%}")


def render_evaluation_results():
    """Render comprehensive evaluation results."""
    st.header("📊 Model Evaluation Results")
    
    metrics = load_metrics()
    report_df = load_classification_report()
    cm_data = load_confusion_matrix()
    
    if not metrics:
        st.info("🔄 Run the pipeline to generate evaluation results")
        return
    
    # Key metrics
    st.subheader("🎯 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        roc_auc = metrics.get('roc_auc', 0)
        st.metric("ROC-AUC", f"{roc_auc:.4f}", 
                 delta="Excellent" if roc_auc > 0.95 else "Good" if roc_auc > 0.85 else "Fair")
    
    # Get metrics from classification report
    if not report_df.empty and '1' in report_df.index:
        class_1_metrics = report_df.loc['1']
        
        with col2:
            precision = class_1_metrics.get('precision', 0)
            st.metric("Precision (Malignant)", f"{precision:.4f}")
        
        with col3:
            recall = class_1_metrics.get('recall', 0)
            st.metric("Recall (Malignant)", f"{recall:.4f}")
        
        with col4:
            f1 = class_1_metrics.get('f1-score', 0)
            st.metric("F1-Score (Malignant)", f"{f1:.4f}")
    
    # Classification report
    if not report_df.empty:
        st.subheader("📋 Detailed Classification Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Format and display the report
            display_df = report_df[report_df.index.isin(['0', '1', 'accuracy', 'macro avg', 'weighted avg'])]
            st.dataframe(display_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True)
        
        with col2:
            # Create a radar chart for class 1 metrics
            if '1' in report_df.index:
                metrics_1 = report_df.loc['1']
                fig = go.Figure()
                
                categories = ['Precision', 'Recall', 'F1-Score']
                values = [
                    metrics_1.get('precision', 0),
                    metrics_1.get('recall', 0),
                    metrics_1.get('f1-score', 0)
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Malignant Class'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Malignant Class Metrics",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    if cm_data and 'matrix' in cm_data:
        st.subheader("🔢 Confusion Matrix")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            matrix = cm_data['matrix']
            labels = cm_data.get('labels', ['Benign', 'Malignant'])
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                text=matrix,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Understanding the Matrix")
            st.markdown("""
            The confusion matrix shows how well the model classifies each category:
            
            - **True Negatives (TN):** Correctly predicted benign cases
            - **False Positives (FP):** Benign cases wrongly predicted as malignant
            - **False Negatives (FN):** Malignant cases wrongly predicted as benign ⚠️
            - **True Positives (TP):** Correctly predicted malignant cases
            
            **For medical diagnosis:**
            - High recall is critical (minimize False Negatives)
            - High precision reduces unnecessary anxiety (minimize False Positives)
            """)


def render_artifacts_status():
    """Render pipeline artifacts status."""
    st.header("📦 Pipeline Artifacts Status")
    
    artifacts = check_artifacts_exist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Trained", "✅ Yes" if artifacts['model_trained'] else "❌ No")
        st.metric("Evaluation Complete", "✅ Yes" if artifacts['evaluation_done'] else "❌ No")
    
    with col2:
        st.metric("Deployment Ready", "✅ Yes" if artifacts['deployment_ready'] else "❌ No")
        st.metric("Monitoring Setup", "✅ Yes" if artifacts['monitoring_setup'] else "❌ No")
    
    with col3:
        # Show file sizes if they exist
        model_path = Path("artifacts/models/best_model.joblib")
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.metric("Model Size", f"{size_mb:.2f} MB")
        
        # Count total artifacts
        artifacts_dir = Path("artifacts")
        if artifacts_dir.exists():
            artifact_count = sum(1 for _ in artifacts_dir.rglob("*") if _.is_file())
            st.metric("Total Artifacts", artifact_count)


def render_documentation():
    """Render model card and other documentation."""
    st.header("📄 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Model Card", "Runbook", "Monitoring Dashboard"])
    
    with tab1:
        model_card = Path("docs/MODEL_CARD.md")
        if model_card.exists():
            st.markdown(model_card.read_text())
        else:
            st.info("Model card will be generated after running the pipeline")
    
    with tab2:
        runbook = Path("docs/RUNBOOK.md")
        if runbook.exists():
            st.markdown(runbook.read_text())
        else:
            st.info("Runbook will be generated after running the pipeline")
    
    with tab3:
        monitoring = Path("docs/MONITORING_DASHBOARD.md")
        if monitoring.exists():
            st.markdown(monitoring.read_text())
        else:
            st.info("Monitoring dashboard will be generated after running the pipeline")


def main() -> None:
    st.set_page_config(
        page_title="ML Pipeline Control Center",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    config = load_config(CONFIG_PATH)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        st.markdown("---")
        
        # Run pipeline button
        st.subheader("Pipeline Execution")
        if st.button("▶️ Run Full Pipeline", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                "Data Ingestion & Validation",
                "Data Splitting",
                "Feature Engineering",
                "Model Training",
                "Evaluation",
                "Packaging",
                "Monitoring Setup"
            ]
            
            for i, stage in enumerate(stages):
                status_text.text(f"Running: {stage}...")
                progress_bar.progress((i + 1) / len(stages))
            
            try:
                run_pipeline(str(CONFIG_PATH))
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Pipeline completed successfully!")
                st.balloons()
                st.rerun()
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"❌ Pipeline failed: {str(e)}")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        artifacts = check_artifacts_exist()
        stages_completed = sum(artifacts.values())
        st.metric("Pipeline Progress", f"{stages_completed}/5 stages")
        
        st.markdown("---")
        
        # Configuration preview
        with st.expander("⚙️ Configuration"):
            st.json({
                "project": config.project.get("name"),
                "version": config.project.get("version"),
                "algorithm": config.training.get("algorithm"),
                "primary_metric": config.experiment.get("primary_metric")
            })
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content
    st.title("🤖 ML Pipeline Control Center")
    st.markdown(f"**{config.project.get('name')}** - Production-Grade Machine Learning Pipeline")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📚 Overview",
        "📋 Project Info", 
        "🔧 Configuration",
        "📊 Evaluation",
        "📦 Artifacts",
        "📄 Documentation"
    ])
    
    with tab1:
        render_pipeline_overview()
    
    with tab2:
        render_project_info(config)
    
    with tab3:
        render_data_insights(config)
        st.markdown("---")
        render_training_config(config)
    
    with tab4:
        render_evaluation_results()
    
    with tab5:
        render_artifacts_status()
    
    with tab6:
        render_documentation()


if __name__ == "__main__":
    main()
