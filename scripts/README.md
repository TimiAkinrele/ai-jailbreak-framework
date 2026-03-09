# Scripts Overview

This folder contains lightweight utilities that support reproducible experiment
operations without changing notebook behavior.

## Retained scripts

- `compact_results_artifacts.py`
  - Compacts secondary outputs into canonical aggregated views.
- `smoke_eval_ood_sets.py`
  - Fast smoke test to confirm OOD evaluation paths are wired correctly.
- `build_ibvs_audit_sample.py`
  - Builds stratified manual IBVS audit packs from processed split data.
- `run_ibvs_component_audit.py`
  - Runs component metrics and tiered evidence audit from annotated samples.

## Cleanup policy

One-off exploratory scripts are removed once their outputs are stabilized in the
main notebook pipeline, to reduce maintenance overhead and repository noise.
