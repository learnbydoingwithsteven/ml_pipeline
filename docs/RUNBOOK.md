# Runbook: Breast Cancer Risk Classifier

## Rollout Steps
1. Deploy container image to staging.
2. Run smoke tests and backfill batch jobs.
3. Perform canary release to 10% traffic for 24h.
4. Promote to production upon guardrail sign-off.

## Rollback Plan
- Trigger rollback job via CI pipeline.
- Revert feature flags to previous model version.
- Notify on-call ML engineer.

## Monitoring & Alerts
- Latency SLO: 200 ms.
- Drift threshold: 0.1.
- Alerts: ml-ops@company.com.
