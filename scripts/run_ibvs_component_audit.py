from __future__ import annotations
"""
Run IBVS manual component audit and tiered evidence audit.

Outputs include:
- per-component PR/F1 against manual labels
- triage utility metrics
- tiered evidence KPIs (T1/T2/T0 + suppression/score-only rates)
- analyst-facing example rows
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

SUPPRESSION_COMPONENTS = [
    "benign_context_suppression",
    "meta_system_discussion_suppression",
    "length_penalty",
]

TIER1_COMPONENTS = [
    "hierarchy_override",
    "role_redefine",
    "system_spoof",
    "tool_directive",
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

TIER2_COMPONENTS = [
    "harm_domain",
    "procedural",
]

MANUAL_BOUNDARY_LABEL_COLS = [
    "gt_hierarchy_override",
    "gt_system_developer_spoof",
    "gt_evasion",
    "gt_tool_misuse_directive",
]

MANUAL_COMPONENT_MAP = {
    "hierarchy_override": "gt_hierarchy_override",
    "system_developer_spoof": "gt_system_developer_spoof",
    "evasion": "gt_evasion",
    "tool_misuse_directive": "gt_tool_misuse_directive",
    "benign_meta_educational_discussion": "gt_benign_meta_educational_discussion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IBVS component audit + triage utility from annotated sample.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Annotated CSV produced by build_ibvs_audit_sample.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for audit metrics.",
    )
    parser.add_argument(
        "--alert-col",
        type=str,
        default="model_alert",
        help="Column used as alert mask for triage utility metrics.",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="If set, skip component PRF when manual labels are missing and emit triage-only outputs.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=10,
        help="How many TP and FP examples to export.",
    )
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    support = int((y_true == 1).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_positive": support,
        "pred_positive": int((y_pred == 1).sum()),
    }


def _component_prediction(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "hierarchy_override":
        return (df["hierarchy_override"].astype(float).to_numpy() > 0.0).astype(int)
    if name == "system_developer_spoof":
        return (df["system_spoof"].astype(float).to_numpy() > 0.0).astype(int)
    if name == "evasion":
        return (df["evasion"].astype(float).to_numpy() > 0.0).astype(int)
    if name == "tool_misuse_directive":
        return (df["tool_directive"].astype(float).to_numpy() > 0.0).astype(int)
    if name == "benign_meta_educational_discussion":
        benign = df["benign_context_suppression"].astype(float).to_numpy() > 0.0
        meta = df["meta_system_discussion_suppression"].astype(float).to_numpy() > 0.0
        return (benign | meta).astype(int)
    raise KeyError(f"Unknown component: {name}")


def compute_component_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for component_name, manual_col in MANUAL_COMPONENT_MAP.items():
        if manual_col not in df.columns:
            raise ValueError(f"Missing required manual label column: {manual_col}")
        sub = df[df[manual_col].notna()].copy()
        if sub.empty:
            continue
        y_true = sub[manual_col].astype(int).to_numpy()
        y_pred = _component_prediction(sub, component_name)
        met = _binary_metrics(y_true, y_pred)
        met["component"] = component_name
        met["manual_label_column"] = manual_col
        met["n_evaluated"] = int(len(sub))
        rows.append(met)
    return pd.DataFrame(rows)


def _has_suppression_only(row: pd.Series) -> int:
    risk_positive = any(float(row.get(c, 0.0)) > 0.0 for c in RISK_COMPONENTS)
    if risk_positive:
        return 0
    if float(row.get("benign_context_suppression", 0.0)) > 0.0:
        return 1
    if float(row.get("meta_system_discussion_suppression", 0.0)) > 0.0:
        return 1
    trig = str(row.get("triggered_rules", "") or "")
    if ("suppressed_unanchored" in trig) or ("meta_system_discussion_suppression" in trig):
        return 1
    return 0


def _split_triggered_rules(v: object) -> List[str]:
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return []
    return [r.strip() for r in s.split(" | ") if r.strip()]


def _classify_rule(rule: str) -> str:
    r = (rule or "").strip().lower()
    if not r:
        return "UNK"
    if (
        r.startswith("hierarchy::")
        or r.startswith("role::")
        or r.startswith("system::")
        or r.startswith("evasion::")
        or r.startswith("tool_risky::")
        or r.startswith("interaction::")
        or r.startswith("anchor::")
    ):
        return "T1"
    if (
        "suppressed_unanchored" in r
        or r == "context::benign_educational_suppression"
        or r == "context::meta_system_discussion_suppression"
    ):
        return "S"
    if (
        r.startswith("harm::")
        or r.startswith("procedural::")
        or r.startswith("tool_generic::")
        or r.startswith("context::")
    ):
        return "T2"
    return "UNK"


def _top_reason(row: pd.Series, cols: List[str]) -> str:
    active = [(c, float(row.get(c, 0.0))) for c in cols if float(row.get(c, 0.0)) > 0.0]
    if not active:
        return "none"
    active.sort(key=lambda x: x[1], reverse=True)
    return active[0][0]


def _build_evidence_summary(row: pd.Series) -> str:
    tier = str(row.get("evidence_tier", "T0"))
    t1_rules = str(row.get("t1_rules", "")).strip()
    t2_rules = str(row.get("t2_rules", "")).strip()
    s_rules = str(row.get("suppression_rules", "")).strip()
    if tier == "T1":
        top = _top_reason(row, TIER1_COMPONENTS)
        if top == "none" and t1_rules:
            top = t1_rules.split(" | ")[0]
        base = f"Actionable evidence: {top}."
        if t1_rules:
            base += f" Signals: {t1_rules}."
    elif tier == "T2":
        top = _top_reason(row, TIER2_COMPONENTS)
        if top == "none" and t2_rules:
            top = t2_rules.split(" | ")[0]
        base = f"Context signals: {top}. No Tier-1 boundary evidence detected."
        if t2_rules:
            base += f" Signals: {t2_rules}."
    else:
        base = "Score-only alert: no IBVS evidence available."
    if s_rules:
        base += f" Suppression applied: {s_rules}."
    return base


def apply_tiered_evidence_framework(df: pd.DataFrame, alert_col: str) -> pd.DataFrame:
    d = df.copy()
    if alert_col not in d.columns:
        raise ValueError(f"Missing alert column: {alert_col}")
    d[alert_col] = d[alert_col].fillna(0).astype(int)

    for c in TIER1_COMPONENTS + TIER2_COMPONENTS + SUPPRESSION_COMPONENTS:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0).astype(float)

    t1_rules_col: List[str] = []
    t2_rules_col: List[str] = []
    s_rules_col: List[str] = []
    unk_rules_col: List[str] = []
    for raw in d.get("triggered_rules", "").tolist():
        rules = _split_triggered_rules(raw)
        t1_rules = sorted({r for r in rules if _classify_rule(r) == "T1"})
        t2_rules = sorted({r for r in rules if _classify_rule(r) == "T2"})
        s_rules = sorted({r for r in rules if _classify_rule(r) == "S"})
        u_rules = sorted({r for r in rules if _classify_rule(r) == "UNK"})
        t1_rules_col.append(" | ".join(t1_rules))
        t2_rules_col.append(" | ".join(t2_rules))
        s_rules_col.append(" | ".join(s_rules))
        unk_rules_col.append(" | ".join(u_rules))

    d["t1_rules"] = t1_rules_col
    d["t2_rules"] = t2_rules_col
    d["suppression_rules"] = s_rules_col
    d["unclassified_rules"] = unk_rules_col

    t1_component_present = d[TIER1_COMPONENTS].gt(0.0).any(axis=1)
    t2_component_present = d[TIER2_COMPONENTS].gt(0.0).any(axis=1)
    suppression_component_present = d[SUPPRESSION_COMPONENTS].gt(0.0).any(axis=1)
    t1_rule_present = d["t1_rules"].str.len().gt(0)
    t2_rule_present = d["t2_rules"].str.len().gt(0)
    suppression_rule_present = d["suppression_rules"].str.len().gt(0)

    t1_present = t1_component_present | t1_rule_present
    t2_present = t2_component_present | t2_rule_present
    suppression_present = suppression_component_present | suppression_rule_present

    # Highest-tier-wins policy:
    # T1 (forensic evidence) > T2 (context) > T0 (score-only)
    d["evidence_tier"] = np.where(t1_present, "T1", np.where(t2_present, "T2", "T0"))
    d["actionable_evidence_flag"] = (d["evidence_tier"] == "T1").astype(int)
    d["suppression_applied_flag"] = suppression_present.astype(int)
    d["score_only_alert_flag"] = ((d[alert_col] == 1) & (d["evidence_tier"] == "T0") & (d["suppression_applied_flag"] == 0)).astype(int)
    d["suppression_only_alert_flag"] = ((d[alert_col] == 1) & (d["evidence_tier"] == "T0") & (d["suppression_applied_flag"] == 1)).astype(int)
    d["evidence_summary"] = d.apply(_build_evidence_summary, axis=1)

    has_labels = all(c in d.columns for c in MANUAL_BOUNDARY_LABEL_COLS)
    if has_labels:
        d["manual_boundary_violation"] = (
            d[MANUAL_BOUNDARY_LABEL_COLS]
            .fillna(0)
            .astype(int)
            .sum(axis=1)
            .gt(0)
            .astype(int)
        )
    else:
        d["manual_boundary_violation"] = np.nan
    return d


def compute_triage_metrics(df: pd.DataFrame, alert_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if alert_col not in df.columns:
        raise ValueError(f"Missing alert column: {alert_col}")
    d = df.copy()
    d[alert_col] = d[alert_col].fillna(0).astype(int)
    if "evidence_tier" in d.columns:
        d["has_risk_evidence"] = d["evidence_tier"].isin(["T1", "T2"]).astype(int)
    else:
        d["has_risk_evidence"] = d[RISK_COMPONENTS].fillna(0.0).gt(0.0).any(axis=1).astype(int)
    d["risk_component_count"] = d[RISK_COMPONENTS].fillna(0.0).gt(0.0).sum(axis=1).astype(int)
    if "suppression_only_alert_flag" in d.columns:
        d["suppression_only"] = d["suppression_only_alert_flag"].astype(int)
    else:
        d["suppression_only"] = d.apply(_has_suppression_only, axis=1).astype(int)
    d["trigger_count"] = d.get("trigger_count", 0).fillna(0).astype(int)

    def _aggregate(sub: pd.DataFrame, key: str) -> Dict[str, float]:
        alerts = sub[sub[alert_col] == 1].copy()
        n_alerts = int(len(alerts))
        if n_alerts == 0:
            return {
                key: "all",
                "n_total": int(len(sub)),
                "n_alerts": 0,
                "explanation_coverage": np.nan,
                "suppression_only_rate": np.nan,
                "actionable_explanation_precision": np.nan,
                "avg_rules_per_alert": np.nan,
                "avg_positive_components_per_alert": np.nan,
            }

        risk_alerts = alerts[alerts["has_risk_evidence"] == 1]
        actionable_precision = (
            float(risk_alerts["label"].astype(int).mean()) if len(risk_alerts) else np.nan
        )
        return {
            key: "all",
            "n_total": int(len(sub)),
            "n_alerts": n_alerts,
            "explanation_coverage": float(alerts["has_risk_evidence"].mean()),
            "suppression_only_rate": float(alerts["suppression_only"].mean()),
            "actionable_explanation_precision": actionable_precision,
            "avg_rules_per_alert": float(alerts["trigger_count"].mean()),
            "avg_positive_components_per_alert": float(alerts["risk_component_count"].mean()),
        }

    overall = pd.DataFrame([_aggregate(d, "ood_set")])

    by_ood_rows: List[Dict[str, float]] = []
    for ood_name, sub in d.groupby("ood_set", dropna=False):
        row = _aggregate(sub, "ood_set")
        row["ood_set"] = ood_name
        by_ood_rows.append(row)
    by_ood = pd.DataFrame(by_ood_rows)
    return overall, by_ood


def compute_tier_metrics(df: pd.DataFrame, alert_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if alert_col not in df.columns:
        raise ValueError(f"Missing alert column: {alert_col}")
    if "evidence_tier" not in df.columns:
        raise ValueError("Tier metrics require evidence_tier columns. Call apply_tiered_evidence_framework first.")

    d = df.copy()
    d[alert_col] = d[alert_col].fillna(0).astype(int)

    def _agg(sub: pd.DataFrame, key: str) -> Dict[str, float]:
        alerts = sub[sub[alert_col] == 1].copy()
        n_alerts = int(len(alerts))
        if n_alerts == 0:
            return {
                key: "all",
                "n_total": int(len(sub)),
                "n_alerts": 0,
                "tier1_alert_count": 0,
                "tier2_alert_count": 0,
                "tier0_alert_count": 0,
                "tier1_evidence_coverage": np.nan,
                "tier1_evidence_precision": np.nan,
                "score_only_rate": np.nan,
                "suppression_only_rate": np.nan,
                "suppression_applied_rate": np.nan,
            }

        t1 = alerts[alerts["evidence_tier"] == "T1"]
        t2 = alerts[alerts["evidence_tier"] == "T2"]
        t0 = alerts[alerts["evidence_tier"] == "T0"]

        if "manual_boundary_violation" in alerts.columns and alerts["manual_boundary_violation"].notna().any():
            t1_precision = float(t1["manual_boundary_violation"].mean()) if len(t1) else np.nan
        else:
            t1_precision = np.nan

        return {
            key: "all",
            "n_total": int(len(sub)),
            "n_alerts": n_alerts,
            "tier1_alert_count": int(len(t1)),
            "tier2_alert_count": int(len(t2)),
            "tier0_alert_count": int(len(t0)),
            "tier1_evidence_coverage": float(len(t1) / n_alerts),
            "tier1_evidence_precision": t1_precision,
            "score_only_rate": float(alerts["score_only_alert_flag"].mean()),
            "suppression_only_rate": float(alerts["suppression_only_alert_flag"].mean()),
            "suppression_applied_rate": float(alerts["suppression_applied_flag"].mean()),
        }

    overall = pd.DataFrame([_agg(d, "ood_set")])
    by_ood_rows: List[Dict[str, float]] = []
    for ood_name, sub in d.groupby("ood_set", dropna=False):
        row = _agg(sub, "ood_set")
        row["ood_set"] = ood_name
        by_ood_rows.append(row)
    by_ood = pd.DataFrame(by_ood_rows)
    return overall, by_ood


def _interpretation(row: pd.Series) -> str:
    if float(row.get("tripwire_alert", 0.0)) > 0:
        return "Tripwire evidence: strong structural boundary-override signal."
    if float(row.get("interaction_system_hierarchy_spoof_chain", 0.0)) > 0:
        return "System+hierarchy spoof chain evidence present."
    if float(row.get("interaction_harm_evasion", 0.0)) > 0:
        return "Harm+evasion interaction evidence present."
    if float(row.get("hierarchy_override", 0.0)) > 0 or float(row.get("system_spoof", 0.0)) > 0:
        return "Boundary override/system-spoof evidence present."
    if _has_suppression_only(row):
        return "Suppression-only evidence: likely benign meta/procedural context."
    if float(row.get("evasion", 0.0)) > 0 or float(row.get("tool_directive", 0.0)) > 0:
        return "Evasion/tool evidence present without high-specific interactions."
    return "Low-signal case; no strong IBVS forensic evidence."


def export_examples(df: pd.DataFrame, out_path: Path, alert_col: str, k: int) -> pd.DataFrame:
    d = df.copy()
    d[alert_col] = d[alert_col].fillna(0).astype(int)
    if "evidence_tier" in d.columns:
        d["has_risk_evidence"] = d["evidence_tier"].isin(["T1", "T2"]).astype(int)
    else:
        d["has_risk_evidence"] = d[RISK_COMPONENTS].fillna(0.0).gt(0.0).any(axis=1).astype(int)
    if "suppression_only_alert_flag" in d.columns:
        d["suppression_only"] = d["suppression_only_alert_flag"].astype(int)
    else:
        d["suppression_only"] = d.apply(_has_suppression_only, axis=1).astype(int)
    d["risk_component_count"] = d[RISK_COMPONENTS].fillna(0.0).gt(0.0).sum(axis=1).astype(int)

    tp = d[(d[alert_col] == 1) & (d["label"].astype(int) == 1) & (d["has_risk_evidence"] == 1)].copy()
    fp = d[(d[alert_col] == 1) & (d["label"].astype(int) == 0)].copy()
    tp = tp.sort_values(["risk_component_count", "trigger_count"], ascending=False).head(k)
    fp = fp.sort_values(["risk_component_count", "trigger_count"], ascending=False).head(k)

    ex = pd.concat([tp.assign(example_type="true_positive_alert"), fp.assign(example_type="false_positive_alert")], ignore_index=True)
    if ex.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ex.to_csv(out_path, index=False)
        return ex

    def _vector_json(row: pd.Series) -> str:
        vec = {c: float(row.get(c, 0.0)) for c in RISK_COMPONENTS + SUPPRESSION_COMPONENTS}
        return json.dumps(vec, ensure_ascii=True)

    ex["prompt_excerpt"] = ex["prompt_text"].astype(str).str.slice(0, 320)
    ex["ibvs_component_vector"] = ex.apply(_vector_json, axis=1)
    ex["analyst_interpretation"] = ex.apply(_interpretation, axis=1)

    keep = [
        "audit_id",
        "example_type",
        "ood_set",
        "label",
        "model_alert",
        "evidence_tier",
        "actionable_evidence_flag",
        "evidence_summary",
        "t1_rules",
        "t2_rules",
        "suppression_rules",
        "prompt_excerpt",
        "ibvs_component_vector",
        "triggered_rules",
        "analyst_interpretation",
    ]
    keep = [c for c in keep if c in ex.columns]
    ex = ex[keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ex.to_csv(out_path, index=False)
    return ex


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    component_out = out_dir / "ibvs_component_precision_metrics.csv"
    triage_overall_out = out_dir / "ibvs_triage_utility_overall.csv"
    triage_by_ood_out = out_dir / "ibvs_triage_utility_by_ood.csv"
    tier_overall_out = out_dir / "ibvs_tier_metrics_overall.csv"
    tier_by_ood_out = out_dir / "ibvs_tier_metrics_by_ood.csv"
    tier_rows_out = out_dir / "ibvs_tiered_audit_rows.csv"
    examples_out = out_dir / "ibvs_triage_examples.csv"
    summary_md_out = out_dir / "ibvs_audit_summary.md"

    can_compute_components = all(c in df.columns and df[c].notna().any() for c in MANUAL_COMPONENT_MAP.values())
    if not can_compute_components and not args.allow_unlabeled:
        missing = [c for c in MANUAL_COMPONENT_MAP.values() if c not in df.columns or not df[c].notna().any()]
        raise ValueError(
            "Component PRF needs manual labels. Missing/empty columns: "
            + ", ".join(missing)
            + ". Re-run with --allow-unlabeled for triage-only metrics."
        )

    tiered_df = apply_tiered_evidence_framework(df, args.alert_col)
    tiered_df.to_csv(tier_rows_out, index=False)

    component_metrics = pd.DataFrame()
    if can_compute_components:
        component_metrics = compute_component_metrics(tiered_df)
        component_metrics.to_csv(component_out, index=False)

    triage_overall, triage_by_ood = compute_triage_metrics(tiered_df, args.alert_col)
    triage_overall.to_csv(triage_overall_out, index=False)
    triage_by_ood.to_csv(triage_by_ood_out, index=False)

    tier_overall, tier_by_ood = compute_tier_metrics(tiered_df, args.alert_col)
    tier_overall.to_csv(tier_overall_out, index=False)
    tier_by_ood.to_csv(tier_by_ood_out, index=False)

    examples = export_examples(tiered_df, examples_out, args.alert_col, args.examples_per_class)

    lines: List[str] = []
    lines.append("# IBVS Audit Summary")
    lines.append("")
    lines.append(f"- Input: `{args.input_csv}`")
    lines.append(f"- Alert column: `{args.alert_col}`")
    lines.append(f"- Rows: {len(df)}")
    lines.append("")
    if not component_metrics.empty:
        lines.append("## Component Precision/Recall/F1")
        lines.append("")
        lines.append(component_metrics.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Component Precision/Recall/F1")
        lines.append("")
        lines.append("Manual component labels not available; triage-only audit generated.")
        lines.append("")
    lines.append("## Triage Utility (Overall)")
    lines.append("")
    lines.append(triage_overall.to_markdown(index=False))
    lines.append("")
    lines.append("## Triage Utility (By OOD Set)")
    lines.append("")
    lines.append(triage_by_ood.to_markdown(index=False))
    lines.append("")
    lines.append("## Tiered Evidence Metrics (Overall)")
    lines.append("")
    lines.append(tier_overall.to_markdown(index=False))
    lines.append("")
    lines.append("## Tiered Evidence Metrics (By OOD Set)")
    lines.append("")
    lines.append(tier_by_ood.to_markdown(index=False))
    lines.append("")
    lines.append(f"- Tiered rows exported: `{tier_rows_out}` ({len(tiered_df)} rows)")
    lines.append("")
    lines.append(f"- Examples exported: `{examples_out}` ({len(examples)} rows)")
    lines.append("")

    summary_md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {triage_overall_out}")
    print(f"Wrote: {triage_by_ood_out}")
    print(f"Wrote: {tier_overall_out}")
    print(f"Wrote: {tier_by_ood_out}")
    print(f"Wrote: {tier_rows_out}")
    print(f"Wrote: {examples_out}")
    print(f"Wrote: {summary_md_out}")
    if not component_metrics.empty:
        print(f"Wrote: {component_out}")
    else:
        print("Skipped component metrics (manual labels unavailable).")


if __name__ == "__main__":
    main()
