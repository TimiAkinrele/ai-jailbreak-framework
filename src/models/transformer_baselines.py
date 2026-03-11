"""Training helpers for supervised transformer text-classification baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tempfile
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _positive_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)[:, 1]


@dataclass(frozen=True)
class TransformerBaselineConfig:
    """Configuration for a single fine-tuned transformer baseline."""

    model_name: str
    output_label: str
    max_length: int = 256
    learning_rate: float = 2e-5
    num_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0


@dataclass
class TransformerBaselineArtifacts:
    """Returned objects and metadata for downstream notebook evaluation."""

    config: TransformerBaselineConfig
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: str
    val_probabilities: np.ndarray
    history: list[dict[str, float]]


class _TokenizedTextDataset(Dataset):
    """Simple fixed-length tokenized dataset for notebook-scale experiments."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        labels: Sequence[int] | None,
        *,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def predict_transformer_probabilities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    device: str | torch.device,
) -> np.ndarray:
    """Run batched inference and return P(y=1) for each text."""

    target_device = torch.device(device)
    dataset = _TokenizedTextDataset(tokenizer, texts, labels=None, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    probabilities: list[np.ndarray] = []
    for batch in loader:
        batch = _to_device(batch, target_device)
        logits = model(**batch).logits
        probabilities.append(_positive_softmax(logits).detach().cpu().numpy())

    return np.concatenate(probabilities, axis=0) if probabilities else np.empty((0,), dtype=float)


def _macro_f1_at_half(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= 0.5).astype(int)
    return float(f1_score(y_true, predictions, average="macro"))


def train_transformer_baseline(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    *,
    config: TransformerBaselineConfig,
    seed: int = 42,
) -> TransformerBaselineArtifacts:
    """Fine-tune a transformer classifier and retain the best validation checkpoint."""

    _set_seed(seed)
    device = _select_device()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    model.to(device)

    train_dataset = _TokenizedTextDataset(
        tokenizer,
        train_texts,
        train_labels,
        max_length=config.max_length,
    )
    val_dataset = _TokenizedTextDataset(
        tokenizer,
        val_texts,
        val_labels,
        max_length=config.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

    classes = np.array([0, 1], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(train_labels))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, float]] = []
    best_val_metric = float("-inf")

    with tempfile.TemporaryDirectory(prefix="ai_jailbreak_tfm_") as temp_dir:
        checkpoint_path = Path(temp_dir) / f"{config.output_label.lower()}_best.pt"

        for epoch_idx in range(config.num_epochs):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                batch = _to_device(batch, device)
                labels = batch.pop("labels")
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                scheduler.step()

                running_loss += float(loss.detach().cpu().item())

            val_probabilities = []
            model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    labels = batch.pop("labels")
                    logits = model(**batch).logits
                    probabilities = _positive_softmax(logits)
                    val_probabilities.append(probabilities.detach().cpu().numpy())
                    batch["labels"] = labels

            val_probabilities_np = (
                np.concatenate(val_probabilities, axis=0) if val_probabilities else np.empty((0,), dtype=float)
            )
            val_macro_f1 = _macro_f1_at_half(np.asarray(val_labels, dtype=int), val_probabilities_np)

            history.append(
                {
                    "epoch": float(epoch_idx + 1),
                    "train_loss": float(running_loss / steps_per_epoch),
                    "val_macro_f1_at_0_5": val_macro_f1,
                }
            )

            if val_macro_f1 >= best_val_metric:
                best_val_metric = val_macro_f1
                torch.save(model.state_dict(), checkpoint_path)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    final_val_probabilities = predict_transformer_probabilities(
        model,
        tokenizer,
        val_texts,
        batch_size=config.eval_batch_size,
        max_length=config.max_length,
        device=device,
    )

    return TransformerBaselineArtifacts(
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=str(device),
        val_probabilities=final_val_probabilities,
        history=history,
    )
