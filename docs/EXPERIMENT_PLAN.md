# Experiment Plan

## Hypothesis
A logistic regression model with standardized features will achieve ≥0.97 ROC-AUC on the validation set.

## Offline Evaluation
- Stratified 60/20/20 train/val/test split.
- 5-fold cross-validation during hyperparameter search.
- Primary metric: ROC-AUC.
- Guardrails: precision ≥0.90, recall ≥0.90.

## Ablations
- Compare logistic regression against random forest baseline.
- Evaluate impact of removing low-variance features.

## Deployment Validation
- Run canary release with 10% traffic.
- Monitor latency and drift metrics for 24 hours before full rollout.
