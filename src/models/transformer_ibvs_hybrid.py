"""Late-fusion helpers for RoBERTa + IBVS decision-layer experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.ibvs import IBVS_V2_NUMERIC_COLUMNS, ibvs_v2_feature_dict


_IBVS_PREFIX = "ibvs2_"
_IBVS_TOTAL_COL = f"{_IBVS_PREFIX}total"
_IBVS_STRUCTURED_COLS: list[str] = [f"{_IBVS_PREFIX}{col}" for col in IBVS_V2_NUMERIC_COLUMNS]


@dataclass(frozen=True)
class RobertaIbvsLateFusionConfig:
    """Configuration for a bounded RoBERTa + IBVS late-fusion experiment."""

    output_label: str
    feature_mode: str
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str | None = "balanced"
    train_score_source: str = "in_sample_roberta_train_scores"

    def __post_init__(self) -> None:
        if self.feature_mode not in {"total", "structured"}:
            raise ValueError("feature_mode must be 'total' or 'structured'.")


@dataclass
class RobertaIbvsLateFusionArtifacts:
    """Fitted meta-classifier and feature metadata for inference/evaluation."""

    config: RobertaIbvsLateFusionConfig
    model: Pipeline
    feature_columns: list[str]


def build_ibvs_feature_frame(texts: Sequence[str]) -> pd.DataFrame:
    """Compute a dense IBVS v2 feature frame for a list of prompts."""

    rows = [ibvs_v2_feature_dict(str(text), prefix=_IBVS_PREFIX) for text in texts]
    df = pd.DataFrame(rows)

    expected_columns = _IBVS_STRUCTURED_COLS + [_IBVS_TOTAL_COL]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df[expected_columns].astype(float)


def build_late_fusion_frame(
    roberta_probabilities: Sequence[float] | np.ndarray,
    ibvs_feature_frame: pd.DataFrame,
    *,
    feature_mode: str,
) -> pd.DataFrame:
    """Combine frozen RoBERTa scores with selected IBVS features."""

    roberta_prob = np.asarray(roberta_probabilities, dtype=float).reshape(-1)
    if len(ibvs_feature_frame) != len(roberta_prob):
        raise ValueError(
            "roberta_probabilities and ibvs_feature_frame must have the same length, "
            f"got {len(roberta_prob)} vs {len(ibvs_feature_frame)}."
        )

    if feature_mode == "total":
        ibvs_cols = [_IBVS_TOTAL_COL]
    elif feature_mode == "structured":
        ibvs_cols = _IBVS_STRUCTURED_COLS + [_IBVS_TOTAL_COL]
    else:
        raise ValueError("feature_mode must be 'total' or 'structured'.")

    out = pd.DataFrame({"roberta_prob": roberta_prob})
    for col in ibvs_cols:
        out[col] = pd.to_numeric(ibvs_feature_frame[col], errors="coerce").fillna(0.0).astype(float).to_numpy()
    return out


def fit_late_fusion_logreg(
    train_roberta_probabilities: Sequence[float] | np.ndarray,
    train_ibvs_feature_frame: pd.DataFrame,
    train_labels: Sequence[int] | np.ndarray,
    *,
    config: RobertaIbvsLateFusionConfig,
) -> RobertaIbvsLateFusionArtifacts:
    """Fit a bounded logistic meta-classifier over RoBERTa score + IBVS features."""

    X_train = build_late_fusion_frame(
        train_roberta_probabilities,
        train_ibvs_feature_frame,
        feature_mode=config.feature_mode,
    )
    y_train = np.asarray(train_labels, dtype=int).reshape(-1)
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train/y_train length mismatch: {len(X_train)} vs {len(y_train)}")

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    return RobertaIbvsLateFusionArtifacts(
        config=config,
        model=model,
        feature_columns=list(X_train.columns),
    )


def predict_late_fusion_probabilities(
    artifacts: RobertaIbvsLateFusionArtifacts,
    roberta_probabilities: Sequence[float] | np.ndarray,
    ibvs_feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Predict P(y=1) for new prompts using the fitted late-fusion model."""

    X_eval = build_late_fusion_frame(
        roberta_probabilities,
        ibvs_feature_frame,
        feature_mode=artifacts.config.feature_mode,
    )
    return artifacts.model.predict_proba(X_eval)[:, 1]
