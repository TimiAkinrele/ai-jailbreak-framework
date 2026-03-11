"""Model helpers for supervised baseline experiments."""

from .transformer_baselines import (
    TransformerBaselineArtifacts,
    TransformerBaselineConfig,
    predict_transformer_probabilities,
    train_transformer_baseline,
)
from .transformer_ibvs_hybrid import (
    RobertaIbvsLateFusionArtifacts,
    RobertaIbvsLateFusionConfig,
    build_ibvs_feature_frame,
    build_late_fusion_frame,
    fit_late_fusion_logreg,
    predict_late_fusion_probabilities,
)

__all__ = [
    "TransformerBaselineArtifacts",
    "TransformerBaselineConfig",
    "predict_transformer_probabilities",
    "train_transformer_baseline",
    "RobertaIbvsLateFusionArtifacts",
    "RobertaIbvsLateFusionConfig",
    "build_ibvs_feature_frame",
    "build_late_fusion_frame",
    "fit_late_fusion_logreg",
    "predict_late_fusion_probabilities",
]
