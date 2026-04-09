# Dissertation Utility Scripts

This directory contains the small utility scripts that support dissertation-facing outputs. The main experiments still live in the notebooks.

## Scripts Relevant To The Dissertation

### `compact_results_artifacts.py`

Purpose:

- builds the canonical aggregate OOD tables under `experiments/results/canonical/`
- keeps the main results folders focused on the files cited in the dissertation

Main canonical outputs:

- `experiments/results/canonical/results_ood_all_models.csv`
- `experiments/results/canonical/ibvs_activation_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_cases_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_trigger_ood_all.csv`

### `build_ibvs_audit_sample.py`

Purpose:

- creates the manual `IBVS` audit sample used for the supplementary review and appendix-style audit work

Main output:

- `experiments/results/audits/ibvs_component_audit_sample_splitA.csv`

### `run_ibvs_component_audit.py`

Purpose:

- scores the manually annotated `IBVS` audit sample
- writes the component-level precision metrics
- writes the triage-utility summaries
- exports analyst-facing example rows

Main outputs:

- `experiments/results/audits/ibvs_component_precision_metrics.csv`
- `experiments/results/audits/ibvs_triage_utility_overall.csv`
- `experiments/results/audits/ibvs_triage_utility_by_ood.csv`
- `experiments/results/audits/ibvs_triage_examples.csv`
- `experiments/results/audits/ibvs_audit_summary.md`

## Scope

This README only documents the scripts that directly support dissertation reporting or dissertation appendix material. The core benchmark logic remains in the notebooks and shared modules under `src/`.
