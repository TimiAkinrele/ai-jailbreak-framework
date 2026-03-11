from __future__ import annotations
"""
Compact secondary result artifacts into canonical aggregate files.

This script is intentionally filesystem-focused:
- build canonical combined tables
- archive secondary per-slice/per-ood diagnostics
"""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
METRICS = RESULTS / "metrics"
IBVS = RESULTS / "ibvs"
CANON = RESULTS / "canonical"
ARCHIVE_ROOT = RESULTS / "archive"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE = ARCHIVE_ROOT / f"compaction_{STAMP}"


def _latest_named_files(search_roots: list[Path], pattern: str) -> list[Path]:
    """Return the newest file for each basename across active + archived roots."""
    latest_by_name: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            current = latest_by_name.get(path.name)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                latest_by_name[path.name] = path
    return sorted(latest_by_name.values(), key=lambda p: p.name)


def _ensure_dirs() -> None:
    CANON.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "metrics").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "ibvs").mkdir(parents=True, exist_ok=True)


def _parse_split_from_name(name: str, token: str = "split") -> str:
    if token not in name:
        return ""
    frag = name.split(token, 1)[1]
    return frag.split("_", 1)[0].split("__", 1)[0]


def _parse_metric_family_from_name(name: str) -> str:
    """Extract the metric family between `metrics_` and `_split`."""
    stem = Path(name).stem
    if not stem.startswith("metrics_") or "_split" not in stem:
        return ""
    return stem[len("metrics_"):].split("_split", 1)[0]


def _build_canonical_metrics_ood() -> int:
    files = _latest_named_files([METRICS, ARCHIVE_ROOT], "metrics_*_split*__ood-*.csv")
    rows: list[pd.DataFrame] = []
    for p in files:
        family = _parse_metric_family_from_name(p.name)
        if family.endswith("_bins") or family.endswith("_slice"):
            continue
        split_tag = _parse_split_from_name(p.name)
        ood_name = p.name.split("__ood-", 1)[1].rsplit(".csv", 1)[0]
        df = pd.read_csv(p)
        if "split" in df.columns:
            df = df[df["split"].astype(str).eq("OOD")].copy()
        if df.empty:
            continue
        if "family" not in df.columns:
            df.insert(0, "family", family)
        else:
            df["family"] = family
        if "split_tag" not in df.columns:
            df.insert(1, "split_tag", split_tag)
        else:
            df["split_tag"] = split_tag
        if "ood_name" not in df.columns:
            df.insert(2, "ood_name", ood_name)
        else:
            df["ood_name"] = df["ood_name"].fillna(ood_name)
        rows.append(df)

    if not rows:
        return 0

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(CANON / "results_ood_all_models.csv", index=False)
    return len(files)


def _build_canonical_ibvs() -> dict[str, int]:
    outputs = {
        "ibvs_activation_ood_all.csv": _latest_named_files([IBVS, ARCHIVE_ROOT], "ibvs_v2_activation_summary_split*__ood-*.csv"),
        "ibvs_fp_trigger_ood_all.csv": _latest_named_files([IBVS, ARCHIVE_ROOT], "ibvs_v2_fp_trigger_summary_split*__ood-*.csv"),
        "ibvs_fp_cases_ood_all.csv": _latest_named_files([IBVS, ARCHIVE_ROOT], "ibvs_v2_fp_cases_split*__ood-*.csv"),
    }
    counts: dict[str, int] = {}

    for out_name, files in outputs.items():
        rows: list[pd.DataFrame] = []
        for p in files:
            split_tag = _parse_split_from_name(p.name)
            ood_name = p.name.split("__ood-", 1)[1].rsplit(".csv", 1)[0]
            df = pd.read_csv(p)
            if "split_tag" not in df.columns:
                df.insert(0, "split_tag", split_tag)
            if "ood_name" not in df.columns:
                df.insert(1, "ood_name", ood_name)
            rows.append(df)

        if rows:
            pd.concat(rows, ignore_index=True).to_csv(CANON / out_name, index=False)
        counts[out_name] = len(files)

    return counts


def _archive_files() -> dict[str, int]:
    # Secondary diagnostics that are safe to archive after canonical views are built.
    metric_patterns = [
        "metrics_*_bins_*.csv",
        "metrics_*_slice_*.csv",
        "metrics_*__ood-*.csv",
        "smoke_metrics_*.csv",
    ]
    ibvs_patterns = [
        "ibvs_v2_activation_summary_split*__ood-*.csv",
        "ibvs_v2_fp_trigger_summary_split*__ood-*.csv",
        "ibvs_v2_fp_cases_split*__ood-*.csv",
        "smoke_ibvs_*.csv",
    ]

    moved_metrics = 0
    moved_ibvs = 0

    for pattern in metric_patterns:
        for p in sorted(METRICS.glob(pattern)):
            target = ARCHIVE / "metrics" / p.name
            if target.exists():
                target = ARCHIVE / "metrics" / f"dup_{p.name}"
            shutil.move(str(p), str(target))
            moved_metrics += 1

    for pattern in ibvs_patterns:
        for p in sorted(IBVS.glob(pattern)):
            target = ARCHIVE / "ibvs" / p.name
            if target.exists():
                target = ARCHIVE / "ibvs" / f"dup_{p.name}"
            shutil.move(str(p), str(target))
            moved_ibvs += 1

    return {"metrics": moved_metrics, "ibvs": moved_ibvs}


def _write_compaction_note(moved: dict[str, int], n_metrics_ood: int, n_ibvs: dict[str, int]) -> None:
    note = CANON / "README_COMPACTION.md"
    text = f"""# Results Compaction

Compaction timestamp: `{STAMP}`
Archive path: `{ARCHIVE.relative_to(ROOT)}`

## Canonical files created
- `experiments/results/canonical/results_ood_all_models.csv`
- `experiments/results/canonical/ibvs_activation_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_trigger_ood_all.csv`
- `experiments/results/canonical/ibvs_fp_cases_ood_all.csv`

## Source file counts used to build canonical files
- OOD metrics files combined: `{n_metrics_ood}`
- IBVS activation files combined: `{n_ibvs.get('ibvs_activation_ood_all.csv', 0)}`
- IBVS FP trigger files combined: `{n_ibvs.get('ibvs_fp_trigger_ood_all.csv', 0)}`
- IBVS FP case files combined: `{n_ibvs.get('ibvs_fp_cases_ood_all.csv', 0)}`

## Files archived (not deleted)
- Metrics files moved: `{moved['metrics']}`
- IBVS files moved: `{moved['ibvs']}`

The archive keeps all secondary diagnostics intact while reducing active-folder clutter.
"""
    note.write_text(text)


def main() -> None:
    _ensure_dirs()
    n_metrics_ood = _build_canonical_metrics_ood()
    n_ibvs = _build_canonical_ibvs()
    moved = _archive_files()
    _write_compaction_note(moved, n_metrics_ood, n_ibvs)

    print("Compaction complete")
    print(f"Archive: {ARCHIVE}")
    print(f"Canonical dir: {CANON}")
    print(f"Moved metrics files: {moved['metrics']}")
    print(f"Moved ibvs files: {moved['ibvs']}")


if __name__ == "__main__":
    main()
