# Canonical Results Tables

This directory holds the canonical aggregate OOD tables used to support the dissertation evaluation chapter.

## Files In This Directory

- `experiments/results/canonical/results_ood_all_models.csv`
- `experiments/results/canonical/ibvs_activation_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_cases_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_trigger_ood_all.csv`

## Why These Files Matter

These tables consolidate cross-OOD outputs into a smaller set of dissertation-facing result files:

- `results_ood_all_models.csv`
  - aggregate OOD model results across the tracked model families
- `ibvs_activation_ood_all.csv`
  - consolidated `IBVS` activation summaries
- `ibvs_fp_cases_ood_all.csv`
  - consolidated false-positive case records
- `ibvs_fp_trigger_ood_all.csv`
  - consolidated false-positive trigger summaries

## How They Are Produced

These files are produced by:

- `scripts/compact_results_artifacts.py`

The compaction step reorganises and aggregates existing result files. It does not retrain models or recalculate the kept split-level results.

## Related Dissertation Files

These canonical tables should be read alongside:

- `experiments/results/metrics/results_table_v2_repeatability.csv`
- `experiments/results/metrics/results_table_difficulty_bins.csv`
- `experiments/results/audits/ibvs_audit_summary.md`
