# Data Contract

- **Producer:** Oncology data warehouse team
- **Consumer:** ML pipeline
- **Schema Version:** 1.0.0
- **Delivery:** Daily batch extract at 02:00 UTC
- **Columns:** As specified in `configs/pipeline.yaml`
- **SLAs:**
  - Freshness ≤ 24 hours
  - Availability ≥ 99%
- **PII Handling:** Dataset contains no direct identifiers. Any PHI must be removed prior to ingestion.
- **Change Management:** Producers must notify consumers 2 weeks prior to schema changes via RFC.
