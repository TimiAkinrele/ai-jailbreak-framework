#!/usr/bin/env python3
"""Smoke test for dual OOD evaluation wiring.

Loads both OOD sets from processed data, samples 50 rows per OOD set,
runs a lightweight semantic baseline (BGE embeddings + logistic regression),
and writes smoke metric/activation outputs for each OOD name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation.eval_metrics import best_threshold_by_macro_f1, evaluate_predictions, results_to_dataframe
from src.features.ibvs import ibvs_v2_with_triggers


def _processed_path(root: Path, split_tag: str) -> Path:
    split_tag = split_tag.upper()
    mapping = {
        "A": "jailbreak_benchmarks_processed_v2.csv",
        "B": "jailbreak_benchmarks_processed_v2_splitB.csv",
        "C": "jailbreak_benchmarks_processed_v2_splitC.csv",
    }
    if split_tag not in mapping:
        raise ValueError("split_tag must be one of A/B/C")
    return root / "data" / "processed" / mapping[split_tag]


def _activation_summary(df_split: pd.DataFrame) -> pd.DataFrame:
    high_cols = [
        "hierarchy_override",
        "system_spoof",
        "interaction_system_hierarchy_spoof_chain",
        "high_specific_risk_anchor",
        "tripwire_alert",
        "harm_domain",
        "evasion",
    ]

    rows = []
    for _, row in df_split.iterrows():
        _, bd, _ = ibvs_v2_with_triggers(str(row["prompt_text"]))
        rows.append({k: float(bd.get(k, 0.0)) for k in high_cols})

    bd_df = pd.DataFrame(rows)
    y = df_split["label"].astype(int).to_numpy()
    pos_mask = y == 1

    out_rows = []
    for col in high_cols:
        vals = pd.to_numeric(bd_df[col], errors="coerce").fillna(0.0)
        out_rows.append(
            {
                "ibvs_component": col,
                "n_positive": int(pos_mask.sum()),
                "activation_rate_on_positive": float((vals[pos_mask] > 0).mean()) if pos_mask.any() else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-tag", default="A", help="A/B/C dataset split tag")
    parser.add_argument("--sample-size", type=int, default=50, help="rows per OOD set for smoke eval")
    args = parser.parse_args()

    root = PROJECT_ROOT
    metrics_dir = root / "experiments" / "results" / "metrics"
    ibvs_dir = root / "experiments" / "results" / "ibvs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ibvs_dir.mkdir(parents=True, exist_ok=True)

    path = _processed_path(root, args.split_tag)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"train", "val", "ood_test", "ood_test_injection", "ood_test_injection_standard"}
    seen = set(df["split"].astype(str).unique())
    missing = required - seen
    if missing:
        raise ValueError(f"Missing required split(s) for smoke run: {sorted(missing)}")

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    X_train = encode(df_train["prompt_text"].astype(str).tolist())
    X_val = encode(df_val["prompt_text"].astype(str).tolist())
    y_train = df_train["label"].astype(int).to_numpy()
    y_val = df_val["label"].astype(int).to_numpy()

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)

    val_score = clf.predict_proba(X_val)[:, 1]
    t_star, _ = best_threshold_by_macro_f1(y_val, val_score, n_grid=1001)

    created = []

    for ood_name in ["ood_test", "ood_test_injection", "ood_test_injection_standard"]:
        d = df[df["split"] == ood_name].copy().reset_index(drop=True)
        n = min(args.sample_size, len(d))
        d = d.sample(n=n, random_state=42).reset_index(drop=True)

        X_ood = encode(d["prompt_text"].astype(str).tolist())
        y_ood = d["label"].astype(int).to_numpy()
        y_score = clf.predict_proba(X_ood)[:, 1]
        y_pred = (y_score >= t_star).astype(int)

        res = evaluate_predictions(
            split_name="OOD",
            y_true=y_ood,
            y_pred=y_pred,
            y_score_for_metrics=y_score,
            print_report=False,
            threshold_note=f"smoke_semantic; ood_name={ood_name}; sample_size={n}",
        )

        mdf = results_to_dataframe("SMOKE_SEM_LOGREG_BGE", [res])
        mdf["eval_track"] = "deployment_threshold"
        mdf["ood_name"] = ood_name

        mpath = metrics_dir / f"smoke_metrics_semantic_split{args.split_tag.upper()}__ood-{ood_name}.csv"
        mdf.to_csv(mpath, index=False)
        created.append(mpath)

        adf = _activation_summary(d)
        adf["ood_name"] = ood_name
        apath = ibvs_dir / f"smoke_ibvs_activation_summary_split{args.split_tag.upper()}__ood-{ood_name}.csv"
        adf.to_csv(apath, index=False)
        created.append(apath)

    for p in created:
        if not p.exists():
            raise RuntimeError(f"Expected output missing: {p}")

    print("Smoke dual-OOD evaluation complete. Outputs:")
    for p in created:
        print(" -", p)


if __name__ == "__main__":
    main()
