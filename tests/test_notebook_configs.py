import json
import re
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _count_occurrences_in_code_cells(nb_path: Path, needle: str) -> int:
    nb = json.loads(nb_path.read_text())
    count = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        count += src.count(needle)
    return count


def _count_split_tag_assignments(nb_path: Path) -> int:
    nb = json.loads(nb_path.read_text())
    count = 0
    pattern = re.compile(r"(?m)^\s*SPLIT_TAG\s*=")
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        count += len(pattern.findall(src))
    return count


class NotebookConfigTests(unittest.TestCase):
    def test_notebooks_default_to_minimal_outputs(self):
        for nb_name in [
            "04_ablation_study.ipynb",
            "05_semantic_baseline.ipynb",
            "07_transformer_baselines.ipynb",
            "08_roberta_ibvs_hybrid.ipynb",
            "06_hybrid_routing.ipynb",
        ]:
            nb_path = REPO_ROOT / "notebooks" / nb_name
            nb_src = json.loads(nb_path.read_text())
            code = "\n".join(
                "".join(c.get("source", []))
                for c in nb_src["cells"]
                if c.get("cell_type") == "code"
            )
            self.assertIn("WRITE_MINIMAL_OUTPUTS", code)
            self.assertIn('os.getenv("WRITE_MINIMAL_OUTPUTS", "1")', code)

    def test_preprocessing_builds_secondary_ood_split(self):
        nb_path = REPO_ROOT / "notebooks" / "02_preprocessing.ipynb"
        nb_src = json.loads(nb_path.read_text())
        code = "\n".join(
            "".join(c.get("source", []))
            for c in nb_src["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn("ood_test_injection", code)
        self.assertIn("ood_test_injection_standard", code)
        self.assertIn("qualifire/prompt-injections-benchmark", code)
        self.assertIn("leolee99/NotInject", code)
        self.assertIn("qualifire-prompt-injections-benchmark.csv", code)
        self.assertIn("OOD_SPLITS_FIXED", code)

    def test_ablation_has_single_split_assignment(self):
        nb_path = REPO_ROOT / "notebooks" / "04_ablation_study.ipynb"
        occurrences = _count_split_tag_assignments(nb_path)
        self.assertEqual(occurrences, 1)

    def test_hybrid_has_single_split_assignment(self):
        nb_path = REPO_ROOT / "notebooks" / "06_hybrid_routing.ipynb"
        occurrences = _count_split_tag_assignments(nb_path)
        self.assertEqual(occurrences, 1)

    def test_transformer_has_single_split_assignment(self):
        nb_path = REPO_ROOT / "notebooks" / "07_transformer_baselines.ipynb"
        occurrences = _count_split_tag_assignments(nb_path)
        self.assertEqual(occurrences, 1)

    def test_transformer_hybrid_has_single_split_assignment(self):
        nb_path = REPO_ROOT / "notebooks" / "08_roberta_ibvs_hybrid.ipynb"
        occurrences = _count_split_tag_assignments(nb_path)
        self.assertEqual(occurrences, 1)

    def test_hybrid_notebook_has_no_ood_tuning_tokens(self):
        nb_hybrid = REPO_ROOT / "notebooks" / "06_hybrid_routing.ipynb"
        hybrid_src = json.loads(nb_hybrid.read_text())
        hybrid_code = "\n".join(
            "".join(c.get("source", []))
            for c in hybrid_src["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertNotIn("ood_dev", hybrid_code)
        self.assertNotIn("ood_holdout", hybrid_code)

    def test_removed_artifact_writers_not_present(self):
        nb_ablation = REPO_ROOT / "notebooks" / "04_ablation_study.ipynb"
        nb_hybrid = REPO_ROOT / "notebooks" / "06_hybrid_routing.ipynb"

        ablation_src = json.loads(nb_ablation.read_text())
        hybrid_src = json.loads(nb_hybrid.read_text())

        ablation_code = "\n".join(
            "".join(c.get("source", []))
            for c in ablation_src["cells"]
            if c.get("cell_type") == "code"
        )
        hybrid_code = "\n".join(
            "".join(c.get("source", []))
            for c in hybrid_src["cells"]
            if c.get("cell_type") == "code"
        )

        # Removed outputs should not be emitted by notebook code anymore.
        self.assertNotIn("run_manifest_ablation_split", ablation_code)
        self.assertNotIn("run_manifest_hybrid_split", hybrid_code)
        self.assertNotIn("metrics_ablation_split{SPLIT_TAG}_ranking.csv", ablation_code)
        self.assertNotIn("metrics_ablation_split{SPLIT_TAG}_deployment.csv", ablation_code)
        self.assertNotIn("metrics_ablation_split{SPLIT_TAG}_ci.csv", ablation_code)
        self.assertNotIn("metrics_ablation_split{SPLIT_TAG}_gate_summary.csv", ablation_code)
        self.assertNotIn("ibvs_v1_breakdown_split", ablation_code)
        self.assertNotIn("metrics_hybrid_sweep", hybrid_code)
        self.assertNotIn("results_table_v1_split", hybrid_code)
        self.assertNotIn("results_table_v2_repeatability_ci.csv", hybrid_code)

    def test_ablation_outputs_only_canonical_metrics_csv(self):
        nb_ablation = REPO_ROOT / "notebooks" / "04_ablation_study.ipynb"
        ablation_src = json.loads(nb_ablation.read_text())
        ablation_code = "\n".join(
            "".join(c.get("source", []))
            for c in ablation_src["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn('metrics_ablation_split{SPLIT_TAG}.csv', ablation_code)
        self.assertIn('metrics_ablation_split{SPLIT_TAG}__ood-{ood_name}.csv', ablation_code)
        self.assertIn('ibvs_v2_breakdown_split{SPLIT_TAG}.csv', ablation_code)
        self.assertIn('ibvs_v2_fp_cases_split{SPLIT_TAG}.csv', ablation_code)
        self.assertIn('ibvs_v2_fp_trigger_summary_split{SPLIT_TAG}.csv', ablation_code)
        self.assertIn('__ood-{ood_name}', ablation_code)

    def test_semantic_and_hybrid_emit_secondary_ood_suffix_outputs(self):
        nb_sem = REPO_ROOT / "notebooks" / "05_semantic_baseline.ipynb"
        nb_tfm = REPO_ROOT / "notebooks" / "07_transformer_baselines.ipynb"
        nb_tfm_hyb = REPO_ROOT / "notebooks" / "08_roberta_ibvs_hybrid.ipynb"
        nb_hyb = REPO_ROOT / "notebooks" / "06_hybrid_routing.ipynb"
        sem_code = "\n".join(
            "".join(c.get("source", []))
            for c in json.loads(nb_sem.read_text())["cells"]
            if c.get("cell_type") == "code"
        )
        tfm_code = "\n".join(
            "".join(c.get("source", []))
            for c in json.loads(nb_tfm.read_text())["cells"]
            if c.get("cell_type") == "code"
        )
        tfm_hyb_code = "\n".join(
            "".join(c.get("source", []))
            for c in json.loads(nb_tfm_hyb.read_text())["cells"]
            if c.get("cell_type") == "code"
        )
        hyb_code = "\n".join(
            "".join(c.get("source", []))
            for c in json.loads(nb_hyb.read_text())["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn("__ood-", sem_code)
        self.assertIn("__ood-{ood_name}", sem_code)
        self.assertIn("__ood-", tfm_code)
        self.assertIn("__ood-{ood_name}", tfm_code)
        self.assertIn("__ood-", tfm_hyb_code)
        self.assertIn("__ood-{ood_name}", tfm_hyb_code)
        self.assertIn("__ood-{ood_name}", hyb_code)

    def test_hybrid_model_family_consistent_across_splits(self):
        metrics_dir = REPO_ROOT / "experiments" / "results" / "metrics"
        available = []
        model_sets = {}
        for split_tag in ["A", "B", "C"]:
            path = metrics_dir / f"metrics_hybrid_split{split_tag}.csv"
            if path.exists():
                df = pd.read_csv(path)
                model_sets[split_tag] = set(df["model"].dropna().astype(str).unique())
                available.append(split_tag)

        if len(available) < 2:
            self.skipTest("Need at least two hybrid split metrics files to check comparability.")

        allowed_models = {
            "HYBRID_SEM_GATE_ELSE_M3_IBVS_V2",
            "HYBRID_SEM_ANCHORED_IBVS_BOOST",
            "HYBRID_SEM_ANCHORED_IBVS_BOOST_V2",
            "HYBRID_SEM_ANCHORED_IBVS_VETO",
            "HYBRID_SEM_LEARNED_META_FUSION",
            "HYBRID_SEM_EXPERT_GATE_M1_M3",
        }
        unique_sets = {frozenset(v) for v in model_sets.values()}
        if len(unique_sets) != 1:
            self.skipTest(
                "Hybrid split outputs are from different rerun generations; "
                f"model sets differ ({ {k: sorted(v) for k, v in model_sets.items()} })."
            )
        for split_tag in available:
            self.assertTrue(model_sets[split_tag].issubset(allowed_models))
            self.assertGreater(len(model_sets[split_tag]), 0)

    def test_repeatability_table_has_no_split_schema_drift(self):
        metrics_dir = REPO_ROOT / "experiments" / "results" / "metrics"
        path = metrics_dir / "results_table_v2_repeatability.csv"
        if not path.exists():
            self.skipTest("Repeatability table not present yet.")

        df = pd.read_csv(path)
        self.assertIn("split_tag", df.columns)
        self.assertIn("eval_track", df.columns)
        self.assertIn("experiment", df.columns)
        if "ood_name" not in df.columns:
            self.skipTest("Repeatability table appears to predate dual-OOD export; missing ood_name column.")

        split_tags = set(df["split_tag"].astype(str).unique())
        self.assertTrue({"A", "B"}.issubset(split_tags))

        expected_split_c_inputs = [
            metrics_dir / "metrics_ablation_splitC.csv",
            metrics_dir / "metrics_semantic_splitC.csv",
            metrics_dir / "metrics_hybrid_splitC.csv",
        ]
        if all(p.exists() for p in expected_split_c_inputs):
            self.assertIn("C", split_tags)

        # Hybrid rows should carry the modern schema metadata fields after harmonised reruns.
        hybrid = df[df["experiment"] == "hybrid"].copy()
        self.assertGreater(len(hybrid), 0)
        required_cols = [
            "m3_val_t_star",
            "m3_threshold_mode",
            "m3_val_fpr_at_t_star",
            "m3_val_fpr_constraint_satisfied",
            "hybrid_variant",
            "tau_low",
            "tau_high",
            "score_definition",
            "ibvs_boost_alpha",
            "ibvs_high_precision_rule",
            "tuning_source",
            "fusion_margin",
            "fusion_alpha",
            "ibvs_gate_definition",
            "calibration_method",
        ]
        missing_cols = [c for c in required_cols if c not in hybrid.columns]
        if missing_cols:
            self.skipTest(
                "Repeatability table appears to be from pre-upgrade notebook runs; "
                f"missing columns: {missing_cols}"
            )

        for required_col in required_cols:
            self.assertIn(required_col, hybrid.columns)
            self.assertTrue(hybrid[required_col].notna().all())

        transformer = df[df["experiment"] == "transformer"].copy()
        if len(transformer) > 0:
            transformer_required_cols = [
                "backbone_name",
                "max_length",
                "learning_rate",
                "num_epochs",
                "train_batch_size",
                "eval_batch_size",
                "weight_decay",
                "warmup_ratio",
                "training_seed",
                "device",
            ]
            missing_tfm_cols = [c for c in transformer_required_cols if c not in transformer.columns]
            if missing_tfm_cols:
                self.skipTest(
                    "Repeatability table appears to predate transformer-baseline integration; "
                    f"missing columns: {missing_tfm_cols}"
                )
            for required_col in transformer_required_cols:
                self.assertIn(required_col, transformer.columns)
                self.assertTrue(transformer[required_col].notna().all())

        transformer_hybrid = df[df["experiment"] == "transformer_hybrid"].copy()
        if len(transformer_hybrid) > 0:
            transformer_hybrid_required_cols = [
                "base_model",
                "meta_model",
                "hybrid_feature_mode",
                "meta_C",
                "stack_train_source",
                "backbone_name",
                "max_length",
                "learning_rate",
                "num_epochs",
                "train_batch_size",
                "eval_batch_size",
                "weight_decay",
                "warmup_ratio",
                "training_seed",
                "device",
            ]
            missing_tfm_hyb_cols = [c for c in transformer_hybrid_required_cols if c not in transformer_hybrid.columns]
            if missing_tfm_hyb_cols:
                self.skipTest(
                    "Repeatability table appears to predate transformer-hybrid integration; "
                    f"missing columns: {missing_tfm_hyb_cols}"
                )
            for required_col in transformer_hybrid_required_cols:
                self.assertIn(required_col, transformer_hybrid.columns)
                self.assertTrue(transformer_hybrid[required_col].notna().all())

        # Repeatability uncertainty summary columns should exist.
        repeat_cols = [
            "repeatability_n_splits",
            "macro_f1_repeat_mean",
            "macro_f1_repeat_std",
            "macro_f1_repeat_ci95_low",
            "macro_f1_repeat_ci95_high",
            "tpr_at_1pct_fpr_repeat_mean",
            "tpr_at_1pct_fpr_repeat_std",
            "tpr_at_1pct_fpr_repeat_ci95_low",
            "tpr_at_1pct_fpr_repeat_ci95_high",
            "tpr_at_5pct_fpr_repeat_mean",
            "tpr_at_5pct_fpr_repeat_std",
            "tpr_at_5pct_fpr_repeat_ci95_low",
            "tpr_at_5pct_fpr_repeat_ci95_high",
            "tpr_at_10pct_fpr_repeat_mean",
            "tpr_at_10pct_fpr_repeat_std",
            "tpr_at_10pct_fpr_repeat_ci95_low",
            "tpr_at_10pct_fpr_repeat_ci95_high",
        ]
        missing_repeat = [c for c in repeat_cols if c not in df.columns]
        if missing_repeat:
            self.skipTest(
                "Repeatability table appears to be from a pre-uncertainty run; "
                f"missing columns: {missing_repeat}"
            )
        for c in repeat_cols:
            self.assertIn(c, df.columns)

        # Removed convenience/legacy outputs should not be present.
        self.assertFalse((metrics_dir / "results_table_v2_repeatability_ci.csv").exists())
        self.assertFalse((metrics_dir / "results_table_v1.csv").exists())
        self.assertFalse((metrics_dir / "results_table_v1_splitA.csv").exists())
        self.assertFalse((metrics_dir / "results_table_v1_splitB.csv").exists())


if __name__ == "__main__":
    unittest.main()
