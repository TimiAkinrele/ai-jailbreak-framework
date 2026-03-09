# AI Jailbreak Classifier

An experimental dissertation repository for prompt-injection and jailbreak detection in large language model inputs.

The project studies whether an interpretable structural feature layer, the **Instruction Boundary Violation Score (IBVS)**, improves detection beyond purely lexical or purely semantic models, especially under out-of-distribution (OOD) evaluation and low-false-positive-rate operating points.

## What This Repository Contains

This repository is organised around a reproducible experiment pipeline:

1. Build a processed dataset from multiple raw jailbreak / benign prompt sources.
2. Engineer lexical, semantic, and structural prompt features.
3. Train and evaluate three model families:
   - ablation models based on TF-IDF, lexical flags, and IBVS
   - a semantic baseline using BGE embeddings and logistic regression
   - semantic-anchored hybrid routing/fusion variants
4. Evaluate the models across repeated ID resamples (`Split A/B/C`) while keeping OOD sets fixed.
5. Export canonical metrics, difficulty-slice analyses, and IBVS forensic artifacts for dissertation reporting.

## Repository Layout

These are the main files and directories that matter when reading the project on Git:

- `README.md`
  - project overview, execution order, and evaluation protocol
- `requirements.txt`
  - lightweight dependency file currently used in the repo
- `data/raw/`
  - raw benchmark inputs used to build the processed dataset
- `data/processed/`
  - processed dataset artifacts used by the notebooks
- `notebooks/`
  - end-to-end experiment notebooks
- `src/`
  - reusable Python modules for IBVS, evaluation, data loading, and shared notebook helpers
- `scripts/`
  - small utilities for audit generation, compaction, and smoke validation
- `tests/`
  - regression tests for evaluation logic, IBVS behavior, and notebook output contracts
- `experiments/results/`
  - generated experiment outputs, metrics tables, IBVS breakdowns, and audit artifacts

## Core Source Modules

- `src/features/ibvs.py`
  - IBVS v2 feature computation, trigger extraction, suppression rules, and tripwire logic
- `src/evaluation/eval_metrics.py`
  - shared evaluation protocol, threshold selection, bootstrap helpers, and metrics table generation
- `src/data/external_ood_loaders.py`
  - helper loaders for secondary OOD datasets
- `src/common/notebook_utils.py`
  - shared notebook utilities such as lexical flags, text statistics, and semantic encoding helpers

## Main Notebooks

- `notebooks/02_preprocessing.ipynb`
  - builds the processed dataset and split metadata
- `notebooks/03_feature_engineering.ipynb`
  - exploratory feature development and sanity-check notebook
- `notebooks/04_ablation_study.ipynb`
  - trains and evaluates the TF-IDF / lexical / IBVS ablation family
- `notebooks/05_semantic_baseline.ipynb`
  - trains and evaluates the semantic baseline (`BAAI/bge-small-en-v1.5` + logistic regression)
- `notebooks/06_hybrid_routing.ipynb`
  - trains and evaluates hybrid routing/fusion models and rebuilds the consolidated comparison tables

## Data and Evaluation Protocol

### In-Distribution (ID)
The in-distribution training pool is built from:

- `AdvBench`
- `JailbreakBench`
- `Deepset prompt injections`
  - benign rows are included directly
  - harmful rows are ratio-capped so the harmful class does not overwhelm ID composition

### OOD Sets
The OOD sets are **evaluation-only** and are kept fixed across `Split A/B/C`.

- `ood_test`
  - harmful: `HarmBench`
  - benign: `Alpaca Instructions`
  - purpose: primary generalisation benchmark for harmful-content-vs-benign behavior

- `ood_test_injection`
  - harmful: `r1char9/prompt-2-prompt-injection-v2-dataset`
  - benign: `leolee99/NotInject`
  - purpose: hard-negative prompt-injection stress test

- `ood_test_injection_standard`
  - harmful + benign: local `qualifire-prompt-injections-benchmark.csv`
  - purpose: balanced prompt-injection benchmark without NotInject-style hard-negative skew

The split composition and counts are recorded in:

- `data/processed/jailbreak_benchmarks_processed_v2_meta.json`

Repeatability resamples are stored as:

- `data/processed/jailbreak_benchmarks_processed_v2.csv` (`Split A`)
- `data/processed/jailbreak_benchmarks_processed_v2_splitB.csv`
- `data/processed/jailbreak_benchmarks_processed_v2_splitC.csv`

## Setup

This repository currently uses a research environment rather than a fully locked dependency spec. `requirements.txt` is not a complete environment file by itself.

A practical minimum setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt pandas numpy scipy scikit-learn xgboost sentence-transformers langdetect jupyter pytest
```

If you already have a working notebook environment, the key libraries used by the pipeline are:

- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `xgboost`
- `sentence-transformers`
- `datasets`
- `langdetect`
- `jupyter`
- `pytest`

## Running the Pipeline

The notebooks are designed to be **artifact-dependent, not session-dependent**:

- notebook `02` must run first when rebuilding processed data
- notebooks `04`, `05`, and `06` can then be run from fresh kernels as long as the required artifacts exist
- notebook `06` also rebuilds cross-model comparison tables from the split-level outputs of `04` and `05`

### 1. Build / Refresh Processed Data

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
```

### 2. Optional Exploratory Feature Notebook

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
```

### 3. Run Split-Level Experiments

Canonical split-level runs:

```bash
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb

SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb

SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb
```

### 4. Full Diagnostics Mode

By default, the experiment notebooks write the minimal canonical outputs. To regenerate the extra per-OOD and bin-level diagnostics, run with `WRITE_MINIMAL_OUTPUTS=0`.

Example:

```bash
WRITE_MINIMAL_OUTPUTS=0 SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
WRITE_MINIMAL_OUTPUTS=0 SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
WRITE_MINIMAL_OUTPUTS=0 SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb
```

### 5. Optional Utility Scripts

- Compact secondary outputs into canonical aggregate tables:

```bash
python scripts/compact_results_artifacts.py
```

- Build the manual IBVS audit sample:

```bash
python scripts/build_ibvs_audit_sample.py
```

- Run the IBVS component and tiered-evidence audit:

```bash
python scripts/run_ibvs_component_audit.py
```

- Smoke-check OOD evaluation wiring:

```bash
python scripts/smoke_eval_ood_sets.py
```

## Canonical Outputs

The core result files produced by the experiment notebooks are:

- `experiments/results/metrics/metrics_ablation_splitA.csv`
- `experiments/results/metrics/metrics_ablation_splitB.csv`
- `experiments/results/metrics/metrics_ablation_splitC.csv`
- `experiments/results/metrics/metrics_semantic_splitA.csv`
- `experiments/results/metrics/metrics_semantic_splitB.csv`
- `experiments/results/metrics/metrics_semantic_splitC.csv`
- `experiments/results/metrics/metrics_hybrid_splitA.csv`
- `experiments/results/metrics/metrics_hybrid_splitB.csv`
- `experiments/results/metrics/metrics_hybrid_splitC.csv`
- `experiments/results/metrics/results_table_v2_repeatability.csv`
- `experiments/results/metrics/results_table_difficulty_bins.csv`

Supporting IBVS forensic outputs include:

- `experiments/results/ibvs/ibvs_v2_breakdown_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_breakdown_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_breakdown_splitC.csv`
- `experiments/results/ibvs/ibvs_v2_fp_cases_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_fp_cases_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_fp_cases_splitC.csv`
- `experiments/results/ibvs/ibvs_v2_fp_trigger_summary_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_fp_trigger_summary_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_fp_trigger_summary_splitC.csv`

Compacted cross-OOD aggregate tables include:

- `experiments/results/canonical/results_ood_all_models.csv`
- `experiments/results/canonical/ibvs_activation_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_cases_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_trigger_ood_all.csv`

Audit artifacts include:

- `experiments/results/audits/ibvs_audit_summary.md`
- `experiments/results/audits/ibvs_component_precision_metrics.csv`
- `experiments/results/audits/ibvs_tier_metrics_overall.csv`
- `experiments/results/audits/ibvs_tier_metrics_by_ood.csv`
- `experiments/results/audits/ibvs_triage_utility_overall.csv`
- `experiments/results/audits/ibvs_triage_utility_by_ood.csv`
- `experiments/results/audits/ibvs_triage_examples.csv`

## Evaluation Metrics

The project uses a security-oriented evaluation protocol rather than relying on a single scalar score.

### Ranking Metrics
These measure score quality without committing to a single deployment threshold.

- `ROC-AUC`
  - threshold-free ranking quality across the full operating range
  - useful for comparing how well models separate harmful from benign prompts overall

- `AUC-PR`
  - area under the precision-recall curve
  - more informative than ROC-AUC when the positive class is relatively sparse or when precision matters

- `TPR@1% FPR`, `TPR@5% FPR`, `TPR@10% FPR`
  - true positive rate measured at fixed false positive rate operating points
  - directly relevant to machine learning security, where low-FPR behavior is often more important than average accuracy

### Thresholded / Deployment Metrics
These measure what happens after selecting a specific operating threshold on validation data.

- `Accuracy`
  - overall classification correctness
  - easy to read, but less informative than low-FPR metrics in safety settings

- `Macro-F1`
  - the unweighted mean of class-wise F1 scores
  - useful when both harmful and benign performance matter and class imbalance exists

- threshold metadata
  - validation threshold value
  - threshold selection mode
  - achieved validation FPR at the selected threshold
  - whether the target FPR constraint was satisfied
  - number of feasible thresholds under the low-FPR constraint

### Hybrid Routing Metrics
These are specific to semantic-anchored hybrid models.

- `semantic_coverage`
  - proportion of prompts decided directly by the semantic model

- `defer_rate`
  - proportion of prompts routed away from the semantic decision path into fallback or fusion logic

These quantities matter because a hybrid model is not just a classifier; it is also a routing policy.

## Evaluation Tracks

The repository distinguishes two complementary evaluation tracks:

- `ranking`
  - threshold-free claims about score quality and low-FPR operating-point behavior
- `deployment_threshold`
  - thresholded operational behavior after selecting the decision threshold on `VAL`

This separation is important in machine learning security research because a model can rank prompts well while still performing poorly at a specific deployment threshold, or vice versa.

## Reproducibility and Validity Rules

The current protocol enforces the following:

1. Thresholds are selected on `VAL`, then frozen for `TEST` and OOD evaluation.
2. OOD sets are never used for training or threshold tuning.
3. Repeatability is evaluated by changing only the ID train/val/test split (`A/B/C`) while keeping OOD rows fixed.
4. Notebook output contracts are regression-tested to reduce accidental schema drift.
5. Shared evaluation logic is centralised in `src/evaluation/eval_metrics.py` so metric definitions stay consistent across notebooks.

## Tests

Run the main regression suite with:

```bash
PYTHONPATH=. pytest -q tests/test_common_notebook_utils.py tests/test_eval_metrics.py tests/test_ibvs.py tests/test_notebook_configs.py
```

These tests cover:

- threshold selector behavior and evaluation invariants
- IBVS v2 feature activations and suppressor behavior
- shared notebook helper utilities
- notebook configuration and output-contract consistency

## Notes

- The repository is notebook-driven by design, but the critical logic is moved into `src/` where possible to reduce duplication.
- If you need a quick script overview, see `scripts/README.md`.
- For dissertation use, the safest tables to cite first are the split-level metrics in `experiments/results/metrics/` and the consolidated tables in `experiments/results/canonical/`.
