# Model Card: Breast Cancer Risk Classifier

**Version:** 1.0.0  \
**Owner:** ml-platform@company.com  \
**Primary Metric:** roc_auc

## Evaluation Summary
- ROC-AUC: 0.995
- Precision (class 1): 0.986
- Recall (class 1): 0.944
- F1 (class 1): 0.965

## Data
- Source: Breast cancer diagnostic features from scikit-learn.
- Rows: 569, Columns: 30 numeric features.

## Ethical Considerations
- Ensure predictions are reviewed by medical professionals.
- Monitor for bias against subpopulations.

## Operational Guidance
- Retrain quarterly or when drift exceeds thresholds.
- Monitor latency, accuracy, and alert channels.
