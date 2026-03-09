from __future__ import annotations
"""
Build a stratified manual IBVS audit sample.

This script is intentionally independent from notebooks so audit packs can be
regenerated reproducibly from processed split files.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval_metrics import best_threshold_low_fpr_with_macro_guard
from src.features.ibvs import IBVS_V2_NUMERIC_COLUMNS, ibvs_v2_with_triggers


RANDOM_SEED = 42
TARGET_FPR = 0.05
MACRO_F1_TOL = 0.02

SPLIT_FILE_BY_TAG = {
    "A": "jailbreak_benchmarks_processed_v2.csv",
    "B": "jailbreak_benchmarks_processed_v2_splitB.csv",
    "C": "jailbreak_benchmarks_processed_v2_splitC.csv",
}

OOD_NAMES = ["ood_test", "ood_test_injection", "ood_test_injection_standard"]

RISK_COMPONENTS = [
    "hierarchy_override",
    "role_redefine",
    "system_spoof",
    "tool_directive",
    "harm_domain",
    "evasion",
    "interaction_hierarchy_system",
    "interaction_system_hierarchy_spoof_chain",
    "interaction_evasion_override",
    "interaction_tool_system",
    "interaction_harm_evasion",
    "interaction_harm_procedural",
    "high_specific_risk_anchor",
    "tripwire_alert",
]

MANUAL_LABEL_COLUMNS = [
    "gt_hierarchy_override",
    "gt_system_developer_spoof",
    "gt_evasion",
    "gt_tool_misuse_directive",
    "gt_benign_meta_educational_discussion",
]

OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"\byou are now\b",
    r"disregard\s+(the\s+)?(previous|above)\s+rules",
    r"\bas an unfiltered model\b",
    r"\bsystem prompt\b",
    r"\bfrom now on\b.*\b(must|will)\b",
]


def _project_root() -> Path:
    return PROJECT_ROOT


def lexical_flags(text: str) -> Dict[str, float]:
    t = (text or "").lower()
    return {
        "has_ignore_prev": float(bool(re.search(OVERRIDE_PATTERNS[0], t))),
        "has_you_are_now": float(bool(re.search(OVERRIDE_PATTERNS[1], t))),
        "has_disregard": float(bool(re.search(OVERRIDE_PATTERNS[2], t))),
        "has_unfiltered": float(bool(re.search(OVERRIDE_PATTERNS[3], t))),
        "has_system_prompt": float(bool(re.search(OVERRIDE_PATTERNS[4], t))),
        "has_from_now_on": float(bool(re.search(OVERRIDE_PATTERNS[5], t))),
        "len_chars": float(len(text or "")),
        "len_tokens_approx": float(len((text or "").split())),
    }


def make_flag_matrix(texts: Iterable[str]) -> csr_matrix:
    flags = pd.DataFrame([lexical_flags(t) for t in texts]).astype(float)
    return csr_matrix(flags.values)


def train_m3_total_and_predict_alerts(df: pd.DataFrame, ood_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    if df_train.empty or df_val.empty:
        raise ValueError("Missing train/val partitions for model-alert generation.")

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    tfidf.fit(df_train["prompt_text"].astype(str))

    X_lex_train = tfidf.transform(df_train["prompt_text"].astype(str))
    X_lex_val = tfidf.transform(df_val["prompt_text"].astype(str))
    X_lex_ood = tfidf.transform(ood_df["prompt_text"].astype(str))

    X_flag_train = make_flag_matrix(df_train["prompt_text"].astype(str))
    X_flag_val = make_flag_matrix(df_val["prompt_text"].astype(str))
    X_flag_ood = make_flag_matrix(ood_df["prompt_text"].astype(str))

    def _ibvs_total_matrix(texts: Sequence[str]) -> csr_matrix:
        totals = []
        for text in texts:
            total, _, _ = ibvs_v2_with_triggers(str(text))
            totals.append(float(total))
        return csr_matrix(np.asarray(totals, dtype=float).reshape(-1, 1))

    X_ibvs_train = _ibvs_total_matrix(df_train["prompt_text"].astype(str).tolist())
    X_ibvs_val = _ibvs_total_matrix(df_val["prompt_text"].astype(str).tolist())
    X_ibvs_ood = _ibvs_total_matrix(ood_df["prompt_text"].astype(str).tolist())

    X_train = hstack([X_lex_train, X_flag_train, X_ibvs_train]).tocsr()
    X_val = hstack([X_lex_val, X_flag_val, X_ibvs_val]).tocsr()
    X_ood = hstack([X_lex_ood, X_flag_ood, X_ibvs_ood]).tocsr()

    y_train = df_train["label"].astype(int).to_numpy()
    y_val = df_val["label"].astype(int).to_numpy()

    # Lightweight alert-context model for audit stratification only.
    # This does not alter training/evaluation notebooks.
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    ood_proba = model.predict_proba(X_ood)[:, 1]
    t_star, _meta = best_threshold_low_fpr_with_macro_guard(
        y_val,
        val_proba,
        target_fpr=TARGET_FPR,
        macro_f1_tolerance=MACRO_F1_TOL,
        enforce_target_fpr=True,
        fallback_to_macro_f1=True,
    )
    ood_alert = (ood_proba >= t_star).astype(int)
    return ood_alert, ood_proba, float(t_star)


def compute_ibvs_columns(df: pd.DataFrame) -> pd.DataFrame:
    totals: List[float] = []
    rows: List[Dict[str, float]] = []
    rules: List[str] = []
    counts: List[int] = []

    for text in df["prompt_text"].astype(str):
        total, breakdown, triggered = ibvs_v2_with_triggers(text)
        totals.append(float(total))
        rows.append(breakdown)
        counts.append(int(len(triggered)))
        rules.append(" | ".join(triggered))

    out = df.copy()
    bd = pd.DataFrame(rows)
    for col in IBVS_V2_NUMERIC_COLUMNS:
        if col not in bd.columns:
            bd[col] = 0.0
    bd = bd[list(IBVS_V2_NUMERIC_COLUMNS)].astype(float)

    out = pd.concat([out.reset_index(drop=True), bd.reset_index(drop=True)], axis=1)
    out["ibvs_v2_total"] = np.asarray(totals, dtype=float)
    out["trigger_count"] = np.asarray(counts, dtype=int)
    out["triggered_rules"] = rules
    out["ibvs_positive_component_count"] = (
        out[RISK_COMPONENTS].gt(0.0).sum(axis=1).astype(int)
    )
    out["ibvs_has_risk_evidence"] = out[RISK_COMPONENTS].gt(0.0).any(axis=1).astype(int)
    return out


def stratified_sample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    strata = df.groupby(["ood_set", "label"], dropna=False).size().reset_index(name="n")
    if strata.empty:
        raise ValueError("No strata found for audit sampling.")

    k = len(strata)
    base = n_total // k
    rem = n_total % k
    strata = strata.sort_values("n", ascending=False).reset_index(drop=True)
    strata["take"] = base
    if rem > 0:
        strata.loc[: rem - 1, "take"] += 1
    strata["take"] = strata[["take", "n"]].min(axis=1).astype(int)

    chunks: List[pd.DataFrame] = []
    for i, r in strata.iterrows():
        mask = (df["ood_set"] == r["ood_set"]) & (df["label"] == r["label"])
        g = df.loc[mask]
        if g.empty or int(r["take"]) <= 0:
            continue
        sample_rs = int(rs.randint(0, 2**31 - 1) + i)
        chunks.append(g.sample(n=int(r["take"]), random_state=sample_rs, replace=False))

    sampled = pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()
    if len(sampled) < min(n_total, len(df)):
        need = min(n_total, len(df)) - len(sampled)
        remaining_idx = df.index.difference(sampled.index)
        if need > 0 and len(remaining_idx) > 0:
            topup = df.loc[remaining_idx].sample(n=min(need, len(remaining_idx)), random_state=seed + 911)
            sampled = pd.concat([sampled, topup], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=seed + 123).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified manual IBVS component audit sample.")
    parser.add_argument("--split-tag", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output path for manual audit CSV.",
    )
    parser.add_argument(
        "--out-scored-full-csv",
        type=Path,
        default=None,
        help="Optional: full scored OOD table for reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _project_root()
    data_path = root / "data" / "processed" / SPLIT_FILE_BY_TAG[args.split_tag]
    if not data_path.exists():
        raise FileNotFoundError(f"Missing split file: {data_path}")

    df = pd.read_csv(data_path)
    ood = df[df["split"].isin(OOD_NAMES)].copy()
    if ood.empty:
        raise ValueError(f"No OOD rows found for {OOD_NAMES}")

    ood = compute_ibvs_columns(ood)
    model_alert, model_score, t_star = train_m3_total_and_predict_alerts(df, ood)
    ood["model_alert"] = model_alert.astype(int)
    ood["model_score"] = model_score.astype(float)
    ood["model_name"] = "M3_TFIDF_PLUS_FLAGS_PLUS_IBVS_V2_TOTAL"
    ood["model_threshold_t_star"] = float(t_star)
    ood["split_tag"] = args.split_tag

    sampled = stratified_sample(ood, args.sample_size, args.seed)
    sampled = sampled.reset_index(drop=True)
    sampled.insert(0, "audit_id", [f"{args.split_tag}-AUD-{i:04d}" for i in range(1, len(sampled) + 1)])

    # Manual annotation fields are intentionally blank for human audit.
    for col in MANUAL_LABEL_COLUMNS:
        sampled[col] = pd.NA
    sampled["analyst_notes"] = ""

    preferred_cols = [
        "audit_id",
        "split_tag",
        "ood_set",
        "dataset_name",
        "source",
        "category",
        "label",
        "model_name",
        "model_threshold_t_star",
        "model_score",
        "model_alert",
        "prompt_text",
        "ibvs_v2_total",
        "trigger_count",
        "ibvs_positive_component_count",
        "ibvs_has_risk_evidence",
    ] + list(IBVS_V2_NUMERIC_COLUMNS) + [
        "triggered_rules",
    ] + MANUAL_LABEL_COLUMNS + [
        "analyst_notes",
    ]
    preferred_cols = [c for c in preferred_cols if c in sampled.columns]
    sampled = sampled[preferred_cols]

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = (
            root
            / "experiments"
            / "results"
            / "audits"
            / f"ibvs_component_audit_sample_split{args.split_tag}.csv"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_csv, index=False)

    if args.out_scored_full_csv is not None:
        args.out_scored_full_csv.parent.mkdir(parents=True, exist_ok=True)
        ood.to_csv(args.out_scored_full_csv, index=False)

    summary = (
        sampled.groupby(["ood_set", "label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["ood_set", "label"])
    )

    print(f"Wrote manual audit sample: {out_csv}")
    print(f"Sample size: {len(sampled)}")
    print(f"Model alert threshold (val t*): {t_star:.6f}")
    print(summary.to_string(index=False))
    print("\nManual label columns to fill:")
    for c in MANUAL_LABEL_COLUMNS:
        print(f" - {c}")


if __name__ == "__main__":
    main()
