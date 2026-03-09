# src/evaluation/eval_metrics.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


def _as_1d_array(name: str, values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """Convert to a non-empty 1D array with a stable dtype."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _validate_same_length(*arrays: np.ndarray, names: Sequence[str]) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        shape_summary = ", ".join(f"{n}={l}" for n, l in zip(names, lengths))
        raise ValueError(f"Input arrays must have equal length: {shape_summary}.")


def _validate_binary_labels(name: str, y: np.ndarray) -> None:
    uniques = np.unique(y)
    if not np.isin(uniques, [0, 1]).all():
        raise ValueError(f"{name} must only contain binary labels 0/1, got {uniques.tolist()}.")


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AP has degenerate but deterministic values for single-class y_true.
    Return them directly to avoid sklearn warnings.
    """
    positives = int((y_true == 1).sum())
    if positives == 0:
        return 0.0
    if positives == len(y_true):
        return 1.0
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC is undefined when only one class is present."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def best_threshold_by_macro_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_grid: int = 1001,
) -> tuple[float, float]:
    """
    Pick threshold t* on VAL to maximise macro-F1.
    Deterministic grid-search over [0,1].
    Returns (t_star, best_macro_f1).
    """
    if n_grid <= 0:
        raise ValueError(f"n_grid must be > 0, got {n_grid}.")

    y_true = _as_1d_array("y_true", y_true, dtype=int)
    y_proba = _as_1d_array("y_proba", y_proba, dtype=float)
    _validate_same_length(y_true, y_proba, names=("y_true", "y_proba"))
    _validate_binary_labels("y_true", y_true)

    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, float(best_f1)


def best_threshold_low_fpr_with_macro_guard(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    target_fpr: float = 0.05,
    macro_f1_tolerance: float = 0.02,
    enforce_target_fpr: bool = False,
    fallback_to_macro_f1: bool = True,
    n_grid: int = 1001,
) -> tuple[float, Dict[str, float]]:
    """
    Pick threshold from VAL using a two-stage rule:
    1) Keep thresholds whose macro-F1 is within `macro_f1_tolerance`
       of the best macro-F1 on VAL.
    2) Among those, maximize TPR with FPR <= `target_fpr`.
       Tie-breakers: lower FPR, then higher macro-F1.

    Optional strict mode:
    - If `enforce_target_fpr=True` and no guarded threshold exists,
      pick the best macro-F1 threshold among those with FPR <= target_fpr.
      This avoids drifting to high-FPR operating points.

    Returns:
      (t_star, diagnostics_dict)
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}.")
    if macro_f1_tolerance < 0.0:
        raise ValueError(f"macro_f1_tolerance must be >= 0, got {macro_f1_tolerance}.")
    if not isinstance(enforce_target_fpr, bool):
        raise TypeError("enforce_target_fpr must be a bool.")
    if not isinstance(fallback_to_macro_f1, bool):
        raise TypeError("fallback_to_macro_f1 must be a bool.")
    if n_grid <= 0:
        raise ValueError(f"n_grid must be > 0, got {n_grid}.")

    y_true = _as_1d_array("y_true", y_true, dtype=int)
    y_proba = _as_1d_array("y_proba", y_proba, dtype=float)
    _validate_same_length(y_true, y_proba, names=("y_true", "y_proba"))
    _validate_binary_labels("y_true", y_true)

    thresholds = np.linspace(0.0, 1.0, n_grid)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        macro = float(f1_score(y_true, y_pred, average="macro"))

        # Binary confusion values with explicit denominators.
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        n_neg = tn + fp
        n_pos = tp + fn
        fpr = float(fp / n_neg) if n_neg > 0 else float("nan")
        tpr = float(tp / n_pos) if n_pos > 0 else float("nan")

        rows.append(
            {
                "threshold": float(t),
                "macro_f1": macro,
                "fpr": fpr,
                "tpr": tpr,
            }
        )

    grid = pd.DataFrame(rows)
    best_macro = float(grid["macro_f1"].max())
    macro_floor = best_macro - float(macro_f1_tolerance)

    finite_fpr = np.isfinite(grid["fpr"].to_numpy())
    feasible_mask = finite_fpr & (grid["fpr"].to_numpy() <= target_fpr)

    feasible = grid.loc[feasible_mask].copy()
    eligible = feasible[feasible["macro_f1"] >= macro_floor].copy()

    if not eligible.empty:
        chosen = eligible.sort_values(
            by=["tpr", "fpr", "macro_f1", "threshold"],
            ascending=[False, True, False, True],
        ).iloc[0]
        selection_mode = "low_fpr_guarded"
    elif enforce_target_fpr and not feasible.empty:
        chosen = feasible.sort_values(
            by=["macro_f1", "tpr", "fpr", "threshold"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selection_mode = "low_fpr_enforced_macro_best"
    else:
        if fallback_to_macro_f1:
            # Fall back to strict macro-F1 optimum if no threshold can satisfy guardrails.
            chosen = grid.sort_values(by=["macro_f1", "threshold"], ascending=[False, True]).iloc[0]
            selection_mode = "macro_f1_fallback"
        elif not feasible.empty:
            chosen = feasible.sort_values(
                by=["macro_f1", "tpr", "fpr", "threshold"],
                ascending=[False, False, True, True],
            ).iloc[0]
            selection_mode = "low_fpr_macro_best_no_guard"
        else:
            chosen = grid.sort_values(
                by=["fpr", "macro_f1", "threshold"],
                ascending=[True, False, True],
            ).iloc[0]
            selection_mode = "closest_fpr_fallback"

    val_fpr = float(chosen["fpr"])
    val_tpr = float(chosen["tpr"])
    min_feasible_fpr = float(np.nanmin(grid["fpr"].to_numpy())) if finite_fpr.any() else float("nan")
    constraint_satisfied = bool(np.isfinite(val_fpr) and val_fpr <= target_fpr)

    return float(chosen["threshold"]), {
        "selection_mode": selection_mode,
        "target_fpr": float(target_fpr),
        "macro_f1_tolerance": float(macro_f1_tolerance),
        "enforce_target_fpr": bool(enforce_target_fpr),
        "fallback_to_macro_f1": bool(fallback_to_macro_f1),
        "val_macro_f1_best": best_macro,
        "val_macro_f1_at_t_star": float(chosen["macro_f1"]),
        "val_fpr_at_t_star": val_fpr,
        "val_tpr_at_t_star": val_tpr,
        "val_fpr_constraint_satisfied": bool(constraint_satisfied),
        "val_fpr_gap_to_target": float(val_fpr - target_fpr) if np.isfinite(val_fpr) else float("nan"),
        "val_fpr_min_possible": min_feasible_fpr,
        "val_num_feasible_thresholds": int(feasible.shape[0]),
    }


def fpr_granularity(y_true: np.ndarray) -> Dict[str, float]:
    """
    Report FPR resolution implied by the number of negative examples.
    This is critical when interpreting very low-FPR operating points (e.g., 1%).
    """
    y_true = _as_1d_array("y_true", y_true, dtype=int)
    _validate_binary_labels("y_true", y_true)
    n_neg = int((y_true == 0).sum())
    return {
        "n_neg": n_neg,
        "fpr_step": float(1.0 / n_neg) if n_neg > 0 else float("nan"),
    }


def _binomial_upper_bound(fp: int, n: int, delta: float) -> float:
    """
    One-sided (1-delta) Wilson upper bound for binomial proportion.
    Chosen to avoid heavyweight runtime deps in evaluation utilities.
    """
    if n <= 0:
        return float("nan")
    if fp < 0 or fp > n:
        raise ValueError(f"fp must satisfy 0 <= fp <= n, got fp={fp}, n={n}.")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    if fp == n:
        return 1.0
    p = float(fp / n)
    z = float(NormalDist().inv_cdf(1.0 - delta))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    radius = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    ub = (center + radius) / denom
    return float(min(max(ub, 0.0), 1.0))


def best_threshold_neyman_pearson(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    target_fpr: float = 0.01,
    delta: float = 0.05,
    fallback_to_macro_f1: bool = True,
) -> tuple[float, Dict[str, float]]:
    """
    Neyman-Pearson style threshold selection:
    choose threshold t* with a one-sided confidence guarantee that
    true FPR <= target_fpr based on VAL negatives.

    Feasibility condition at threshold t:
      upper_confidence_bound(FPR_t) <= target_fpr
    where the upper bound is exact Clopper-Pearson.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}.")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    if not isinstance(fallback_to_macro_f1, bool):
        raise TypeError("fallback_to_macro_f1 must be a bool.")

    y_true = _as_1d_array("y_true", y_true, dtype=int)
    y_proba = _as_1d_array("y_proba", y_proba, dtype=float)
    _validate_same_length(y_true, y_proba, names=("y_true", "y_proba"))
    _validate_binary_labels("y_true", y_true)

    thresholds = np.unique(np.clip(y_proba, 0.0, 1.0))
    thresholds = np.unique(np.concatenate(([0.0], thresholds, [1.0])))

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        macro = float(f1_score(y_true, y_pred, average="macro"))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        n_neg = tn + fp
        n_pos = tp + fn
        fpr = float(fp / n_neg) if n_neg > 0 else float("nan")
        tpr = float(tp / n_pos) if n_pos > 0 else float("nan")
        fpr_ub = _binomial_upper_bound(fp, n_neg, delta) if n_neg > 0 else float("nan")
        rows.append(
            {
                "threshold": float(t),
                "macro_f1": macro,
                "fpr": fpr,
                "tpr": tpr,
                "n_neg": int(n_neg),
                "fp": int(fp),
                "fpr_upper_bound": float(fpr_ub),
            }
        )

    grid = pd.DataFrame(rows)
    feasible = grid[np.isfinite(grid["fpr_upper_bound"]) & (grid["fpr_upper_bound"] <= target_fpr)].copy()

    if not feasible.empty:
        chosen = feasible.sort_values(
            by=["tpr", "macro_f1", "fpr_upper_bound", "threshold"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selection_mode = "np_feasible_best_tpr"
    else:
        if fallback_to_macro_f1:
            chosen = grid.sort_values(by=["macro_f1", "threshold"], ascending=[False, True]).iloc[0]
            selection_mode = "np_macro_f1_fallback"
        else:
            chosen = grid.sort_values(
                by=["fpr_upper_bound", "tpr", "macro_f1", "threshold"],
                ascending=[True, False, False, True],
            ).iloc[0]
            selection_mode = "np_min_upper_bound_fallback"

    n_neg = int(chosen["n_neg"])
    fpr_step = float(1.0 / n_neg) if n_neg > 0 else float("nan")
    return float(chosen["threshold"]), {
        "selection_mode": selection_mode,
        "target_fpr": float(target_fpr),
        "delta": float(delta),
        "confidence": float(1.0 - delta),
        "fallback_to_macro_f1": bool(fallback_to_macro_f1),
        "val_macro_f1_at_t_star": float(chosen["macro_f1"]),
        "val_fpr_at_t_star": float(chosen["fpr"]),
        "val_tpr_at_t_star": float(chosen["tpr"]),
        "val_fpr_upper_bound_at_t_star": float(chosen["fpr_upper_bound"]),
        "val_fpr_constraint_satisfied": bool(
            np.isfinite(chosen["fpr_upper_bound"]) and chosen["fpr_upper_bound"] <= target_fpr
        ),
        "val_num_feasible_thresholds": int(feasible.shape[0]),
        "val_n_neg": int(n_neg),
        "val_fpr_step": fpr_step,
    }


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """
    Return the best (maximum) TPR achievable with FPR <= target_fpr
    using ROC curve points derived from (y_true, y_score).
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}.")

    y_true = _as_1d_array("y_true", y_true, dtype=int)
    y_score = _as_1d_array("y_score", y_score, dtype=float)
    _validate_same_length(y_true, y_score, names=("y_true", "y_score"))
    _validate_binary_labels("y_true", y_true)

    if np.unique(y_true).size < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def tpr_at_fprs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fprs: Sequence[float] = (0.01, 0.05, 0.10),
) -> Dict[str, float]:
    """
    Compute TPR@FPR for multiple operating points.
    Default: 1%, 5%, 10% FPR.
    """
    out = {}
    for f in fprs:
        out[f"tpr_at_{int(f*100)}pct_fpr"] = tpr_at_fpr(y_true, y_score, target_fpr=f)
    return out


def _bootstrap_ci_from_samples(samples: np.ndarray, ci_level: float) -> tuple[float, float]:
    if samples.size == 0:
        return float("nan"), float("nan")
    alpha = (100.0 - ci_level) / 2.0
    lo = float(np.percentile(samples, alpha))
    hi = float(np.percentile(samples, 100.0 - alpha))
    return lo, hi


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci_level: float = 95.0,
) -> Dict[str, float]:
    """
    Bootstrap CI for a metric computed from (y_true, y_score).
    Returns a compact metadata contract for notebook/table use.
    """
    if n_boot <= 0:
        raise ValueError(f"n_boot must be > 0, got {n_boot}.")
    if not 0.0 < ci_level < 100.0:
        raise ValueError(f"ci_level must be in (0, 100), got {ci_level}.")

    y_true = _as_1d_array("y_true", y_true, dtype=int)
    y_score = _as_1d_array("y_score", y_score, dtype=float)
    _validate_same_length(y_true, y_score, names=("y_true", "y_score"))
    _validate_binary_labels("y_true", y_true)

    rng = np.random.default_rng(seed)
    n = y_true.size
    point_estimate = float(metric_fn(y_true, y_score))

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draw_val = float(metric_fn(y_true[idx], y_score[idx]))
        if np.isfinite(draw_val):
            draws.append(draw_val)

    draws_arr = np.asarray(draws, dtype=float)
    ci_low, ci_high = _bootstrap_ci_from_samples(draws_arr, ci_level=ci_level)
    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def bootstrap_delta_ci(
    y_true_baseline: np.ndarray,
    y_score_baseline: np.ndarray,
    y_true_candidate: np.ndarray,
    y_score_candidate: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci_level: float = 95.0,
    paired_if_possible: bool = True,
) -> Dict[str, float]:
    """
    Bootstrap CI for delta metric:
      delta = metric(candidate) - metric(baseline)
    """
    if n_boot <= 0:
        raise ValueError(f"n_boot must be > 0, got {n_boot}.")
    if not 0.0 < ci_level < 100.0:
        raise ValueError(f"ci_level must be in (0, 100), got {ci_level}.")

    y_true_baseline = _as_1d_array("y_true_baseline", y_true_baseline, dtype=int)
    y_score_baseline = _as_1d_array("y_score_baseline", y_score_baseline, dtype=float)
    y_true_candidate = _as_1d_array("y_true_candidate", y_true_candidate, dtype=int)
    y_score_candidate = _as_1d_array("y_score_candidate", y_score_candidate, dtype=float)
    _validate_same_length(
        y_true_baseline,
        y_score_baseline,
        names=("y_true_baseline", "y_score_baseline"),
    )
    _validate_same_length(
        y_true_candidate,
        y_score_candidate,
        names=("y_true_candidate", "y_score_candidate"),
    )
    _validate_binary_labels("y_true_baseline", y_true_baseline)
    _validate_binary_labels("y_true_candidate", y_true_candidate)

    baseline_point = float(metric_fn(y_true_baseline, y_score_baseline))
    candidate_point = float(metric_fn(y_true_candidate, y_score_candidate))
    point_estimate = float(candidate_point - baseline_point)

    rng = np.random.default_rng(seed)
    n_base = y_true_baseline.size
    n_cand = y_true_candidate.size
    paired = bool(
        paired_if_possible
        and n_base == n_cand
        and np.array_equal(y_true_baseline, y_true_candidate)
    )

    draws = []
    for _ in range(n_boot):
        if paired:
            idx = rng.integers(0, n_base, size=n_base)
            delta_draw = float(
                metric_fn(y_true_candidate[idx], y_score_candidate[idx])
                - metric_fn(y_true_baseline[idx], y_score_baseline[idx])
            )
        else:
            idx_base = rng.integers(0, n_base, size=n_base)
            idx_cand = rng.integers(0, n_cand, size=n_cand)
            delta_draw = float(
                metric_fn(y_true_candidate[idx_cand], y_score_candidate[idx_cand])
                - metric_fn(y_true_baseline[idx_base], y_score_baseline[idx_base])
            )

        if np.isfinite(delta_draw):
            draws.append(delta_draw)

    draws_arr = np.asarray(draws, dtype=float)
    ci_low, ci_high = _bootstrap_ci_from_samples(draws_arr, ci_level=ci_level)
    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


@dataclass
class EvalResult:
    split: str
    acc: float
    macro_f1: float
    auc_pr: float
    roc_auc: float
    tpr_at_1pct_fpr: float
    tpr_at_5pct_fpr: float
    tpr_at_10pct_fpr: float

    # Hybrid-only optional fields
    semantic_coverage: Optional[float] = None
    defer_rate: Optional[float] = None

    # Notes: thresholding / score definition etc.
    threshold_note: Optional[str] = None


def evaluate_predictions(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_for_metrics: np.ndarray,
    *,
    print_report: bool = True,
    route: Optional[Sequence[str]] = None,
    threshold_note: Optional[str] = None,
) -> EvalResult:
    """
    Standard evaluation used by notebooks 04/05/06.

    IMPORTANT:
    - y_score_for_metrics must be ONE consistent score vector.
      Examples:
        * Notebook 04 (M1/M2/M3): model predicted proba for class 1
        * Notebook 05 (Semantic): semantic predicted proba for class 1
        * Notebook 06 (Hybrid): unified hybrid score
          (semantic score outside uncertainty region, fallback score inside)
    """

    y_true = _as_1d_array("y_true", y_true, dtype=int)
    y_pred = _as_1d_array("y_pred", y_pred, dtype=int)
    y_score_for_metrics = _as_1d_array("y_score_for_metrics", y_score_for_metrics, dtype=float)
    _validate_same_length(
        y_true, y_pred, y_score_for_metrics,
        names=("y_true", "y_pred", "y_score_for_metrics"),
    )
    _validate_binary_labels("y_true", y_true)
    _validate_binary_labels("y_pred", y_pred)
    if not np.isfinite(y_score_for_metrics).all():
        raise ValueError("y_score_for_metrics must not contain NaN/inf.")

    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))

    auc_pr = _safe_average_precision(y_true, y_score_for_metrics)
    roc_auc = _safe_roc_auc(y_true, y_score_for_metrics)

    tprs = tpr_at_fprs(y_true, y_score_for_metrics, fprs=(0.01, 0.05, 0.10))

    if print_report:
        print(f"\n=== {split_name} ===")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)
        print(f"AUC-PR:  {auc_pr:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(
            "TPR @ FPR: "
            f"1%={tprs['tpr_at_1pct_fpr']:.4f}, "
            f"5%={tprs['tpr_at_5pct_fpr']:.4f}, "
            f"10%={tprs['tpr_at_10pct_fpr']:.4f}"
        )

    semantic_coverage = None
    defer_rate = None

    if route is not None:
        if isinstance(route, str):
            raise TypeError("route must be a sequence of route labels, not a single string.")
        r = pd.Series(list(route))
        if len(r) != len(y_true):
            raise ValueError(
                "route length must match y_true length, "
                f"got route={len(r)} and y_true={len(y_true)}."
            )
        route_props = r.value_counts(normalize=True).to_dict()
        if print_report:
            print("Routing proportions:", {k: round(v, 4) for k, v in route_props.items()})
        defer_rate = float(r.astype(str).str.startswith("fallback_").mean())
        semantic_coverage = float(1.0 - defer_rate)

    return EvalResult(
        split=split_name,
        acc=acc,
        macro_f1=macro,
        auc_pr=auc_pr,
        roc_auc=roc_auc,
        tpr_at_1pct_fpr=float(tprs["tpr_at_1pct_fpr"]),
        tpr_at_5pct_fpr=float(tprs["tpr_at_5pct_fpr"]),
        tpr_at_10pct_fpr=float(tprs["tpr_at_10pct_fpr"]),
        semantic_coverage=semantic_coverage,
        defer_rate=defer_rate,
        threshold_note=threshold_note,
    )


def results_to_dataframe(model_name: str, results: Sequence[EvalResult]) -> pd.DataFrame:
    """
    Convert EvalResult objects into a tidy DataFrame for dissertation tables.
    """
    cols = [
        "model", "split",
        "acc", "macro_f1", "auc_pr", "roc_auc",
        "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "tpr_at_10pct_fpr",
        "semantic_coverage", "defer_rate",
        "threshold_note",
    ]

    rows = []
    for r in results:
        d = asdict(r)
        d["model"] = model_name
        rows.append(d)

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    return df[cols]
