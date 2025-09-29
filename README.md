# ML Pipeline Demo

Comprehensive, end-to-end machine learning pipeline with governance-ready artifacts and a control center UI.

## Features
- ✅ Problem framing artifacts (PRD, data sheet, experiment plan).
- ✅ Data ingestion, validation, and split management with reproducible config.
- ✅ Feature engineering with scaling and variance-based selection.
- ✅ Hyperparameter-tuned model training with guardrail quality gates.
- ✅ Automated evaluation, model card generation, and deployment manifest.
- ✅ Monitoring simulation and dashboard export.
- ✅ Streamlit UI for running the pipeline and exploring outputs.
- ✅ Pytest smoke test covering the full DAG.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline from the CLI:

```bash
python main.py
```

Launch the control center UI:

```bash
streamlit run ui/app.py
```

## Project Structure

```
.
├── configs/              # YAML configuration for the pipeline
├── docs/                 # Governance artifacts (PRD, data sheet, runbook, model card)
├── src/ml_pipeline/      # Pipeline modules for each lifecycle stage
├── tests/                # Pytest smoke tests
├── ui/                   # Streamlit control center
├── artifacts/            # Generated models, metrics, and manifests
└── main.py               # CLI entry point
```

## Quality Gates
- Data validation ensures schema, null ratios, and non-negative constraints.
- Experiment guardrails enforce precision and recall thresholds alongside ROC-AUC.
- Packaging stage emits deployment manifest and runbook for governance.
- Monitoring stage simulates drift detection and publishes dashboards.

## Next Steps
- Integrate with feature store and real-time monitoring backend.
- Add CI workflows for automated retraining and deployment.
