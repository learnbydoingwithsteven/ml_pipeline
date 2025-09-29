# Data Sheet: Breast Cancer Diagnostic Dataset

## Motivation
Support development of a diagnostic aid that classifies breast tumors as malignant or benign.

## Composition
- 569 patient records collected from digitized FNA of breast masses.
- 30 continuous features extracted from images.
- Target label: 0 (benign), 1 (malignant).

## Collection Process
Dataset is bundled with scikit-learn and derived from the UCI Machine Learning Repository.

## Preprocessing
- No missing values.
- Features standardized during feature engineering.
- Negative values disallowed per validation rules.

## Uses
- Binary classification research and prototyping.
- Not a substitute for medical diagnosis; for educational use in this repo.

## Distribution
- Freely available via scikit-learn, license CC BY 4.0.

## Maintenance
- Retrain quarterly or upon drift alert.
- Log data lineage via pipeline artifacts and config versioning.
