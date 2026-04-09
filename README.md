# AI Jailbreak Classifier

This repository contains the dissertation experiment pipeline for jailbreak and prompt-injection detection in large language model prompts.

The central methodological idea is to compare lexical, structural, semantic, transformer, and hybrid model families under a common in-distribution training setup and multiple held-out OOD regimes. The main structural contribution is the **Instruction Boundary Violation Score** (`IBVS`).

## Dissertation Research Questions

The dissertation investigates the following research questions:

1. To what extent can simple lexical signals detect jailbreak and prompt-injection attacks in LLM prompts?
2. Does the proposed `IBVS` provide additional detection capability beyond lexical baselines?
3. How do interpretable lexical and structural models compare with modern transformer encoder baselines for jailbreak detection?
4. Can structural prompt-boundary signals provide useful complementary information when combined with strong neural encoders?
5. How does the effectiveness of these approaches vary under differing OOD evaluation regimes?

## What Is In Scope

The repository is organised around the dissertation workflow:

1. Build the processed benchmark from tracked raw files plus the external OOD loaders.
2. Run the model ladder notebooks.
3. Export split-level metrics, consolidated tables, IBVS forensic outputs, and manual-audit outputs.

This is a notebook-first research repository. The most important directories are:

- `data/raw/`
- `data/processed/`
- `notebooks/`
- `src/`
- `experiments/results/`

## Data Files

### Raw files tracked in the repository

These are the local inputs used in preprocessing:

- `data/raw/advbench-harmful_behaviors.csv`
- `data/raw/alpaca_instructions.csv`
- `data/raw/deepset_prompt_injections.csv`
- `data/raw/harmbench_behaviors_text_all.csv`
- `data/raw/jbb-benign-behaviors.csv`
- `data/raw/jbb-harmful-behaviors.csv`
- `data/raw/qualifire-prompt-injections-benchmark.csv`

### External loader-backed OOD sources

Two preprocessing inputs are not stored as local raw CSV files in `data/raw/`. They are loaded through the external OOD loader in `src/data/external_ood_loaders.py`:

- `r1char9/prompt-2-prompt-injection-v2-dataset`
  - attack side of `ood_test_injection`
- `leolee99/NotInject`
  - benign hard-negative side of `ood_test_injection`

So the local raw files are not the entire preprocessing input set. The hard-negative injection OOD benchmark depends on these two external loader-backed sources as well as the tracked local files.

### Processed benchmark files

The tracked processed benchmark is:

- `data/processed/jailbreak_benchmarks_processed_v2.csv`
- `data/processed/jailbreak_benchmarks_processed_v2_meta.json`
- `data/processed/jailbreak_benchmarks_processed_v2_splitB.csv`
- `data/processed/jailbreak_benchmarks_processed_v2_splitB_meta.json`
- `data/processed/jailbreak_benchmarks_processed_v2_splitC.csv`
- `data/processed/jailbreak_benchmarks_processed_v2_splitC_meta.json`

These files correspond to the methodology chapter’s processed benchmark and repeated `Split A`, `Split B`, and `Split C` setup.

## Notebook Map

These notebooks are the dissertation-facing workflow:

- `notebooks/02_preprocessing.ipynb`
  - builds the processed benchmark and fixed OOD sets
- `notebooks/03_feature_engineering.ipynb`
  - exploratory feature inspection
- `notebooks/04_ablation_study.ipynb`
  - TF-IDF, lexical flags, and `IBVS` ablation family
- `notebooks/05_semantic_baseline.ipynb`
  - `BAAI/bge-small-en-v1.5` semantic baseline with logistic regression
- `notebooks/06_hybrid_routing.ipynb`
  - semantic-centred hybrid routing and fusion variants
- `notebooks/07_transformer_baselines.ipynb`
  - RoBERTa-base and DeBERTa-base baselines
- `notebooks/08_roberta_ibvs_hybrid.ipynb`
  - RoBERTa + `IBVS` late-fusion experiment

## Source Modules

The main reusable modules behind the notebooks are:

- `src/common/notebook_utils.py`
- `src/data/external_ood_loaders.py`
- `src/evaluation/eval_metrics.py`
- `src/features/ibvs.py`
- `src/models/transformer_baselines.py`
- `src/models/transformer_ibvs_hybrid.py`

These files hold the shared logic for preprocessing support, `IBVS`, evaluation, transformer baselines, and the RoBERTa + `IBVS` hybrid.

## How To Run

### 1. Rebuild the processed benchmark

Run preprocessing first if you want to rebuild the benchmark from raw inputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
```

This step uses:

- the tracked local raw files under `data/raw/`
- the local Qualifire file
- the external loader-backed `r1char9` and `NotInject` datasets

### 2. Run the main experiment notebooks

The main dissertation experiment path is:

```bash
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/07_transformer_baselines.ipynb

SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb
SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/07_transformer_baselines.ipynb

SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/05_semantic_baseline.ipynb
SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/06_hybrid_routing.ipynb
SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/07_transformer_baselines.ipynb
```

These notebooks write the split-level result files used throughout the dissertation.

### 3. Run the transformer-hybrid extension

Run this only if you want the RoBERTa + `IBVS` late-fusion comparison:

```bash
SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/08_roberta_ibvs_hybrid.ipynb
SPLIT_TAG=B jupyter nbconvert --to notebook --execute --inplace notebooks/08_roberta_ibvs_hybrid.ipynb
SPLIT_TAG=C jupyter nbconvert --to notebook --execute --inplace notebooks/08_roberta_ibvs_hybrid.ipynb
```

### 4. Optional full diagnostics mode

The notebooks default to minimal active outputs. If you want the extra OOD-suffixed and bin-level diagnostics, set `WRITE_MINIMAL_OUTPUTS=0`.

Example:

```bash
WRITE_MINIMAL_OUTPUTS=0 SPLIT_TAG=A jupyter nbconvert --to notebook --execute --inplace notebooks/04_ablation_study.ipynb
```

## Dissertation File Map

### Methodology and dataset design

The methodology chapter maps most directly to:

- `data/processed/jailbreak_benchmarks_processed_v2_meta.json`
- `notebooks/02_preprocessing.ipynb`
- `src/data/external_ood_loaders.py`
- `src/features/ibvs.py`
- `src/evaluation/eval_metrics.py`

### Main evaluation chapter

The main evaluation chapter maps most directly to:

- `experiments/results/metrics/results_table_v2_repeatability.csv`
- `experiments/results/metrics/results_table_difficulty_bins.csv`
- `experiments/results/canonical/results_ood_all_models.csv`
- `experiments/results/canonical/ibvs_activation_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_cases_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_trigger_ood_all.csv`

Split-level support files are:

- `experiments/results/metrics/metrics_ablation_splitA.csv`
- `experiments/results/metrics/metrics_ablation_splitB.csv`
- `experiments/results/metrics/metrics_ablation_splitC.csv`
- `experiments/results/metrics/metrics_semantic_splitA.csv`
- `experiments/results/metrics/metrics_semantic_splitB.csv`
- `experiments/results/metrics/metrics_semantic_splitC.csv`
- `experiments/results/metrics/metrics_transformer_splitA.csv`
- `experiments/results/metrics/metrics_transformer_splitB.csv`
- `experiments/results/metrics/metrics_transformer_splitC.csv`
- `experiments/results/metrics/metrics_hybrid_splitA.csv`
- `experiments/results/metrics/metrics_hybrid_splitB.csv`
- `experiments/results/metrics/metrics_hybrid_splitC.csv`
- `experiments/results/metrics/metrics_transformer_hybrid_splitA.csv`
- `experiments/results/metrics/metrics_transformer_hybrid_splitB.csv`
- `experiments/results/metrics/metrics_transformer_hybrid_splitC.csv`

### IBVS forensics and false-positive review

The `IBVS` review sections map to:

- `experiments/results/ibvs/ibvs_v2_breakdown_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_breakdown_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_breakdown_splitC.csv`
- `experiments/results/ibvs/ibvs_v2_activation_summary_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_activation_summary_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_activation_summary_splitC.csv`
- `experiments/results/ibvs/ibvs_v2_fp_cases_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_fp_cases_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_fp_cases_splitC.csv`
- `experiments/results/ibvs/ibvs_v2_fp_trigger_summary_splitA.csv`
- `experiments/results/ibvs/ibvs_v2_fp_trigger_summary_splitB.csv`
- `experiments/results/ibvs/ibvs_v2_fp_trigger_summary_splitC.csv`

### Appendix diagnostics

The appendix material maps to:

- `experiments/results/metrics/results_table_difficulty_bins.csv`
- `experiments/results/audits/ibvs_component_audit_sample_splitA.csv`
- `experiments/results/audits/ibvs_component_precision_metrics.csv`
- `experiments/results/audits/ibvs_triage_utility_overall.csv`
- `experiments/results/audits/ibvs_triage_utility_by_ood.csv`
- `experiments/results/audits/ibvs_triage_examples.csv`
- `experiments/results/audits/ibvs_audit_summary.md`

## Notes

- The canonical loader for external OOD data is `src/data/external_ood_loaders.py`.
- The active ablation path now reflects the dissertation ladder: `TF-IDF`, `TF-IDF + flags`, `IBVS v1 Total`, `IBVS v2 Total`, and `IBVS v2 Structured`.
- The README files in `scripts/` and `experiments/results/canonical/` describe only the dissertation-facing utilities and result tables.
