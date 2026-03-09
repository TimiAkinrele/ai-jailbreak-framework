import unittest

try:
    import numpy as np
    from src.evaluation.eval_metrics import (
        best_threshold_low_fpr_with_macro_guard,
        best_threshold_neyman_pearson,
        bootstrap_metric_ci,
        bootstrap_delta_ci,
        fpr_granularity,
        tpr_at_fpr,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    np = None
    best_threshold_low_fpr_with_macro_guard = None
    best_threshold_neyman_pearson = None
    bootstrap_metric_ci = None
    bootstrap_delta_ci = None
    fpr_granularity = None
    tpr_at_fpr = None
    _IMPORT_ERROR = exc


@unittest.skipIf(np is None, f"Missing test dependency: {_IMPORT_ERROR}")
class ThresholdSelectionTests(unittest.TestCase):
    def test_guarded_selection_when_eligible_exists(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        y_proba = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=float)

        _, meta = best_threshold_low_fpr_with_macro_guard(
            y_true,
            y_proba,
            target_fpr=0.05,
            macro_f1_tolerance=0.05,
            enforce_target_fpr=True,
            fallback_to_macro_f1=True,
            n_grid=1001,
        )

        self.assertEqual(meta["selection_mode"], "low_fpr_guarded")
        self.assertTrue(meta["val_fpr_constraint_satisfied"])
        self.assertLessEqual(meta["val_fpr_at_t_star"], 0.05)

    def test_enforced_selection_when_guarded_infeasible(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        y_proba = np.array([0.82, 0.84, 0.86, 0.9, 0.88, 0.9, 0.92, 0.95], dtype=float)

        _, meta = best_threshold_low_fpr_with_macro_guard(
            y_true,
            y_proba,
            target_fpr=0.05,
            macro_f1_tolerance=0.001,
            enforce_target_fpr=True,
            fallback_to_macro_f1=True,
            n_grid=1001,
        )

        self.assertEqual(meta["selection_mode"], "low_fpr_enforced_macro_best")
        self.assertTrue(meta["val_fpr_constraint_satisfied"])
        self.assertLessEqual(meta["val_fpr_at_t_star"], 0.05)

    def test_fallback_when_no_fpr_defined(self):
        # No negatives -> FPR undefined across all thresholds.
        y_true = np.array([1, 1, 1, 1], dtype=int)
        y_proba = np.array([0.1, 0.3, 0.7, 0.9], dtype=float)

        _, meta = best_threshold_low_fpr_with_macro_guard(
            y_true,
            y_proba,
            target_fpr=0.05,
            macro_f1_tolerance=0.02,
            enforce_target_fpr=True,
            fallback_to_macro_f1=True,
            n_grid=101,
        )

        self.assertEqual(meta["selection_mode"], "macro_f1_fallback")
        self.assertFalse(meta["val_fpr_constraint_satisfied"])

    def test_np_threshold_feasible_mode_and_metadata(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
        y_proba = np.array([0.05, 0.1, 0.12, 0.2, 0.22, 0.7, 0.75, 0.8, 0.85, 0.9], dtype=float)

        _, meta = best_threshold_neyman_pearson(
            y_true,
            y_proba,
            target_fpr=0.40,
            delta=0.10,
            fallback_to_macro_f1=True,
        )

        self.assertEqual(meta["selection_mode"], "np_feasible_best_tpr")
        self.assertTrue(meta["val_fpr_constraint_satisfied"])
        self.assertGreaterEqual(meta["val_num_feasible_thresholds"], 1)
        self.assertEqual(meta["val_n_neg"], 5)
        self.assertAlmostEqual(meta["val_fpr_step"], 0.2)

    def test_np_threshold_fallback_when_no_feasible(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        y_proba = np.array([0.9, 0.88, 0.86, 0.84, 0.92, 0.93, 0.94, 0.95], dtype=float)

        _, meta = best_threshold_neyman_pearson(
            y_true,
            y_proba,
            target_fpr=0.0,
            delta=0.05,
            fallback_to_macro_f1=True,
        )

        self.assertEqual(meta["selection_mode"], "np_macro_f1_fallback")
        self.assertFalse(meta["val_fpr_constraint_satisfied"])

    def test_fpr_granularity_reports_negative_count_and_step(self):
        y_true = np.array([0, 0, 1, 1, 1], dtype=int)
        meta = fpr_granularity(y_true)
        self.assertEqual(meta["n_neg"], 2)
        self.assertAlmostEqual(meta["fpr_step"], 0.5)


@unittest.skipIf(np is None, f"Missing test dependency: {_IMPORT_ERROR}")
class BootstrapMetricsTests(unittest.TestCase):
    def test_bootstrap_metric_ci_is_reproducible(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0], dtype=int)
        y_score = np.array([0.1, 0.2, 0.3, 0.55, 0.6, 0.7, 0.9, 0.25, 0.8, 0.05], dtype=float)

        out1 = bootstrap_metric_ci(
            y_true,
            y_score,
            metric_fn=lambda yt, ys: tpr_at_fpr(yt, ys, target_fpr=0.05),
            n_boot=200,
            seed=123,
        )
        out2 = bootstrap_metric_ci(
            y_true,
            y_score,
            metric_fn=lambda yt, ys: tpr_at_fpr(yt, ys, target_fpr=0.05),
            n_boot=200,
            seed=123,
        )

        self.assertEqual(out1["n_boot"], 200)
        self.assertEqual(out1["seed"], 123)
        self.assertAlmostEqual(out1["point_estimate"], out2["point_estimate"])
        self.assertAlmostEqual(out1["ci_low"], out2["ci_low"])
        self.assertAlmostEqual(out1["ci_high"], out2["ci_high"])
        self.assertLessEqual(out1["ci_low"], out1["ci_high"])

    def test_bootstrap_delta_ci_contract(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0], dtype=int)
        baseline = np.array([0.1, 0.2, 0.35, 0.5, 0.55, 0.65, 0.85, 0.3, 0.75, 0.1], dtype=float)
        candidate = np.array([0.08, 0.18, 0.28, 0.58, 0.63, 0.74, 0.92, 0.22, 0.83, 0.06], dtype=float)

        out = bootstrap_delta_ci(
            y_true_baseline=y_true,
            y_score_baseline=baseline,
            y_true_candidate=y_true,
            y_score_candidate=candidate,
            metric_fn=lambda yt, ys: tpr_at_fpr(yt, ys, target_fpr=0.05),
            n_boot=200,
            seed=99,
        )

        self.assertEqual(out["n_boot"], 200)
        self.assertEqual(out["seed"], 99)
        self.assertTrue(np.isfinite(out["point_estimate"]))
        self.assertLessEqual(out["ci_low"], out["ci_high"])


if __name__ == "__main__":
    unittest.main()
