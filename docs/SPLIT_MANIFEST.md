# Split Manifest

- **Train:** 60% of dataset, stratified by target.
- **Validation:** 20% of dataset, stratified by target.
- **Test:** 20% of dataset, held out for final evaluation.

Splits are deterministically generated using random state 42. See `train_validation_test_split` in `src/ml_pipeline/data_management.py`.
