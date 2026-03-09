from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'datasets' package is required for external OOD loaders. "
        "Install with: pip install datasets"
    ) from exc


PREFERRED_SPLITS = ("train", "test", "validation")
NOTINJECT_CONFIGS = ("NotInject_one", "NotInject_two", "NotInject_three")
QUALIFIRE_DATASET_ID = "qualifire/prompt-injections-benchmark"
QUALIFIRE_FALLBACK_DATASET_ID = "r1char9/prompt-2-prompt-injection-v2-dataset"
DEFAULT_QUALIFIRE_LOCAL_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "qualifire-prompt-injections-benchmark.csv"
)
DEFAULT_QUALIFIRE_LOCAL_JSONL = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "qualifire" / "prompt_injections_benchmark.jsonl"
)


@dataclass(frozen=True)
class LoadedSplit:
    split_name: str
    frame: pd.DataFrame
    config_name: str | None = None


def _to_dataframe(ds_obj: Dataset) -> pd.DataFrame:
    frame = ds_obj.to_pandas()
    if frame.empty:
        raise ValueError("Loaded dataset split is empty.")
    return frame.reset_index(drop=True)


def _hf_token() -> str | None:
    # Optional token support for private/gated sources, while still working
    # with public datasets without any auth.
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _load_dataset_with_optional_token(dataset_id: str, *, config_name: str | None = None):
    token = _hf_token()
    kwargs = {"token": token} if token else {}
    if config_name is not None:
        kwargs["name"] = config_name
    return load_dataset(dataset_id, **kwargs)


def first_available_split(dataset_id: str, config_name: str | None = None) -> LoadedSplit:
    """Load a dataset and return the first preferred split.

    Priority order: train -> test -> validation.
    Falls back to the single available split if none of the preferred split
    names are present. Raises a clear error when no splits are available.
    """
    try:
        ds_any = _load_dataset_with_optional_token(dataset_id, config_name=config_name)
    except Exception as exc:
        cfg = f" (config={config_name})" if config_name else ""
        raise RuntimeError(
            f"Failed to load dataset '{dataset_id}'{cfg}. "
            "If this dataset is gated/private, set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
            "in your environment and ensure your account has access."
        ) from exc

    if isinstance(ds_any, Dataset):
        return LoadedSplit(split_name="dataset", frame=_to_dataframe(ds_any), config_name=config_name)

    if not isinstance(ds_any, DatasetDict):
        raise TypeError(f"Unsupported dataset type from {dataset_id}: {type(ds_any)!r}")

    split_names = list(ds_any.keys())
    if not split_names:
        raise ValueError(f"No splits found for dataset: {dataset_id}")

    for split_name in PREFERRED_SPLITS:
        if split_name in ds_any:
            return LoadedSplit(
                split_name=split_name,
                frame=_to_dataframe(ds_any[split_name]),
                config_name=config_name,
            )

    # Fallback to the first available split when preferred names do not exist.
    split_name = split_names[0]
    return LoadedSplit(
        split_name=split_name,
        frame=_to_dataframe(ds_any[split_name]),
        config_name=config_name,
    )


def _first_existing_column(columns: Iterable[str], preferred: Iterable[str], dataset_id: str) -> str:
    cols = set(columns)
    for c in preferred:
        if c in cols:
            return c
    raise ValueError(
        f"Could not find any supported text column for {dataset_id}. "
        f"Tried={list(preferred)}; available={sorted(cols)}"
    )


def _resolve_local_qualifire_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Local Qualifire file not found: {candidate}")
        return candidate

    if DEFAULT_QUALIFIRE_LOCAL_CSV.exists():
        return DEFAULT_QUALIFIRE_LOCAL_CSV
    if DEFAULT_QUALIFIRE_LOCAL_JSONL.exists():
        return DEFAULT_QUALIFIRE_LOCAL_JSONL

    raise FileNotFoundError(
        "Local Qualifire file not found. Expected one of:\n"
        f" - {DEFAULT_QUALIFIRE_LOCAL_CSV}\n"
        f" - {DEFAULT_QUALIFIRE_LOCAL_JSONL}"
    )


def _normalize_qualifire_frame(frame: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    required = {"text", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Local Qualifire file must contain columns ['text', 'label']; missing={missing}. "
            f"Found={list(frame.columns)}"
        )

    labels = frame["label"].astype(str).str.strip().str.lower()
    mapped = labels.map({"jailbreak": 1, "benign": 0})
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].unique().tolist())
        raise ValueError(
            "Local Qualifire labels must be {'jailbreak','benign'}. "
            f"Unexpected labels={bad[:10]}"
        )

    out = pd.DataFrame(
        {
            "prompt_text": frame["text"].astype(str),
            "label": mapped.astype(int),
            "dataset_name": "qualifire_local",
            "attack_family": labels.where(labels.eq("jailbreak"), "benign"),
            "source_id": frame.index.astype(int),
            "source": "qualifire",
            "category": labels,
            "text": frame["text"].astype(str),
            "hf_split_name": "local",
            "hf_dataset_id": "qualifire/prompt-injections-benchmark (local)",
            "hf_config_name": source_file.name,
        }
    )
    return out


def load_qualifire_local(path: str | Path | None = None) -> pd.DataFrame:
    """Load local Qualifire benchmark file with strict schema validation.

    Accepts CSV or JSONL with required columns: text,label where
    label is one of {'jailbreak','benign'}.
    """
    source_file = _resolve_local_qualifire_path(path)
    suffix = source_file.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(source_file)
    elif suffix == ".jsonl":
        frame = pd.read_json(source_file, lines=True)
    else:
        raise ValueError(
            f"Unsupported local Qualifire format: {source_file}. "
            "Use .csv or .jsonl"
        )

    return _normalize_qualifire_frame(frame, source_file)


def load_qualifire_standard(seed: int = 42, path: str | Path | None = None) -> pd.DataFrame:
    """Balanced Qualifire-only injection OOD mix: jailbreak vs benign."""
    df = load_qualifire_local(path=path)
    df_attack = df[df["label"] == 1].copy()
    df_benign = df[df["label"] == 0].copy()

    n = min(len(df_attack), len(df_benign))
    if n <= 0:
        raise ValueError(
            "Local Qualifire file must contain both benign and jailbreak rows. "
            f"counts={{1: {len(df_attack)}, 0: {len(df_benign)}}}"
        )

    df_attack = df_attack.sample(n=n, random_state=seed)
    df_benign = df_benign.sample(n=n, random_state=seed)
    out = pd.concat([df_attack, df_benign], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def load_qualifire_jailbreak_attacks(seed: int = 42) -> pd.DataFrame:
    """Load Qualifire prompt-injection benchmark and keep only jailbreak rows.

    The preferred source is `qualifire/prompt-injections-benchmark`. If it is
    gated/unavailable without authentication, we transparently fall back to the
    public `r1char9/prompt-2-prompt-injection-v2-dataset` dataset to preserve
    a no-login pipeline and keep OOD-injection disjoint from ID sources.
    """
    candidate_ids = [QUALIFIRE_DATASET_ID, QUALIFIRE_FALLBACK_DATASET_ID]
    load_errors: list[str] = []
    loaded: LoadedSplit | None = None
    dataset_id: str | None = None

    for candidate in candidate_ids:
        try:
            loaded = first_available_split(candidate)
            dataset_id = candidate
            break
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")

    if loaded is None or dataset_id is None:
        raise RuntimeError(
            "Unable to load any attack dataset for OOD injection set. "
            f"Tried={candidate_ids}. Errors={' | '.join(load_errors[:3])}"
        )

    frame = loaded.frame.copy()

    if dataset_id == QUALIFIRE_FALLBACK_DATASET_ID:
        text_col = _first_existing_column(
            frame.columns,
            ["prompt_injection", "prompt", "text", "instruction", "input", "query"],
            dataset_id,
        )
        # This dataset is attack-only for our usage; treat all rows as label=1.
        attack_mask = frame[text_col].astype(str).str.len() > 0
    else:
        text_col = _first_existing_column(
            frame.columns,
            ["text", "prompt", "instruction", "input", "query"],
            dataset_id,
        )
        label_col = _first_existing_column(frame.columns, ["label", "labels"], dataset_id)

        labels_raw = frame[label_col].astype(str).str.strip().str.lower()
        mapped = labels_raw.map({"jailbreak": 1, "benign": 0})

        # Fallback for integer-like labels.
        if mapped.isna().all():
            mapped = pd.to_numeric(frame[label_col], errors="coerce")

        attack_mask = mapped == 1
        if int(attack_mask.sum()) == 0:
            raise ValueError(
                "Qualifire loader found zero jailbreak rows. "
                f"Unique raw labels: {sorted(set(labels_raw.dropna().tolist()))[:10]}"
            )

    attack_df = frame.loc[attack_mask].copy().reset_index(drop=True)
    if dataset_id == QUALIFIRE_DATASET_ID:
        source_name = "qualifire"
        attack_family = "jailbreak"
    elif dataset_id == QUALIFIRE_FALLBACK_DATASET_ID:
        source_name = "r1char9_prompt_injection_v2"
        attack_family = "jailbreak"
    else:
        source_name = "prompt_injection_attacks"
        attack_family = "jailbreak"

    out = pd.DataFrame(
        {
            "prompt_text": attack_df[text_col].astype(str),
            "label": 1,
            "dataset_name": source_name,
            "attack_family": attack_family,
            "source_id": attack_df.index.astype(int),
            "hf_split_name": loaded.split_name,
            "hf_dataset_id": dataset_id,
            "hf_config_name": loaded.config_name,
            # Requested convenience aliases.
            "source": source_name,
            "category": attack_family,
            "text": attack_df[text_col].astype(str),
        }
    )

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _resolve_notinject_configs(dataset_id: str) -> list[str]:
    token = _hf_token()
    kwargs = {"token": token} if token else {}
    try:
        available = get_dataset_config_names(dataset_id, **kwargs)
    except Exception:
        available = []

    if not available:
        return list(NOTINJECT_CONFIGS)

    selected = [cfg for cfg in NOTINJECT_CONFIGS if cfg in available]
    return selected if selected else list(available)


def load_notinject_benign(seed: int = 42) -> pd.DataFrame:
    """Load NotInject hard negatives as label=0 across all known configs."""
    dataset_id = "leolee99/NotInject"
    blocks: list[pd.DataFrame] = []
    failures: list[str] = []

    def _make_block(frame: pd.DataFrame, *, hf_split_name: str, hf_config_name: str | None) -> pd.DataFrame:
        text_col = _first_existing_column(
            frame.columns,
            ["text", "prompt", "instruction", "input", "query"],
            f"{dataset_id}:{hf_split_name}",
        )
        block = pd.DataFrame(
            {
                "prompt_text": frame[text_col].astype(str),
                "label": 0,
                "dataset_name": "notinject",
                "attack_family": "hard_negative",
                "source_id": frame.index.astype(int),
                "hf_split_name": hf_split_name,
                "hf_dataset_id": dataset_id,
                "hf_config_name": hf_config_name,
                # Requested convenience aliases.
                "source": "notinject",
                "category": "hard_negative",
                "text": frame[text_col].astype(str),
            }
        )

        trigger_candidates = [
            "trigger_tokens",
            "triggers",
            "keywords",
            "trigger_words",
            "token_list",
            "word_list",
        ]
        for col in trigger_candidates:
            if col in frame.columns:
                block["trigger_tokens"] = frame[col].astype(str)
                break
        return block

    # Pattern 1: dataset exposes NotInject variants as split names directly.
    try:
        ds_direct = _load_dataset_with_optional_token(dataset_id)
        if isinstance(ds_direct, DatasetDict):
            split_keys = list(ds_direct.keys())
            target_splits = [s for s in NOTINJECT_CONFIGS if s in split_keys]
            if target_splits:
                for split_name in target_splits:
                    frame = _to_dataframe(ds_direct[split_name])
                    blocks.append(
                        _make_block(frame, hf_split_name=split_name, hf_config_name=None)
                    )
    except Exception as exc:
        failures.append(f"direct-splits: {exc}")

    # Pattern 2: dataset exposes variants as configs.
    if not blocks:
        configs = _resolve_notinject_configs(dataset_id)
        for cfg in configs:
            try:
                loaded = first_available_split(dataset_id, config_name=cfg)
                frame = loaded.frame.copy()
            except Exception as exc:
                failures.append(f"{cfg}: {exc}")
                continue

            blocks.append(
                _make_block(frame, hf_split_name=loaded.split_name, hf_config_name=cfg)
            )

    if not blocks:
        joined_failures = "; ".join(failures[:5]) if failures else "none"
        raise RuntimeError(
            "Unable to load any NotInject config. "
            f"Failures={joined_failures}"
        )

    out = pd.concat(blocks, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out
