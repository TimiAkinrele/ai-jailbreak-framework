"""Pure helpers shared across notebooks to reduce duplicated experiment code."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


DEFAULT_OVERRIDE_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"\byou are now\b",
    r"disregard\s+(the\s+)?(previous|above)\s+rules",
    r"\bas an unfiltered model\b",
    r"\bsystem prompt\b",
    r"\bfrom now on\b.*\b(must|will)\b",
)


def get_git_commit(repo_root: Path) -> str:
    """Return the current git commit hash or 'unknown' when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def lexical_flags(
    text: str,
    override_patterns: Sequence[str] = DEFAULT_OVERRIDE_PATTERNS,
) -> dict[str, float | bool]:
    """Build the lightweight lexical flag set used by M2/M3 style notebooks."""
    raw_text = text or ""
    lowered = raw_text.lower()
    return {
        "has_ignore_prev": bool(re.search(override_patterns[0], lowered)),
        "has_you_are_now": bool(re.search(override_patterns[1], lowered)),
        "has_disregard": bool(re.search(override_patterns[2], lowered)),
        "has_unfiltered": bool(re.search(override_patterns[3], lowered)),
        "has_system_prompt": bool(re.search(override_patterns[4], lowered)),
        "has_from_now_on": bool(re.search(override_patterns[5], lowered)),
        "len_chars": len(raw_text),
        "len_tokens_approx": len(raw_text.split()),
    }


def make_flag_matrix(
    prompt_texts: Iterable[str] | pd.Series | pd.DataFrame,
    override_patterns: Sequence[str] = DEFAULT_OVERRIDE_PATTERNS,
) -> csr_matrix:
    """Convert prompt texts into the sparse lexical-flag matrix used by notebooks."""
    if isinstance(prompt_texts, pd.DataFrame):
        if "prompt_text" not in prompt_texts.columns:
            raise ValueError("Expected a 'prompt_text' column when passing a DataFrame.")
        texts = prompt_texts["prompt_text"].astype(str).tolist()
    elif isinstance(prompt_texts, pd.Series):
        texts = prompt_texts.astype(str).tolist()
    else:
        texts = [str(text) for text in prompt_texts]

    flags_df = pd.DataFrame(
        [lexical_flags(text, override_patterns=override_patterns) for text in texts]
    )
    return csr_matrix(flags_df.values.astype(float))


def encode_texts(model, texts: Sequence[str], batch_size: int = 32, show_progress_bar: bool = False):
    """Wrap sentence-transformer encoding with the shared normalization settings."""
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def text_stats(prompt_series: pd.Series) -> pd.DataFrame:
    """Compute token-count and lexical-complexity features for bin-level diagnostics."""
    tok_counts: list[float] = []
    uniq_counts: list[float] = []
    avg_tok_lens: list[float] = []
    ttrs: list[float] = []

    for text in prompt_series.astype(str):
        toks = [token for token in re.findall(r"[A-Za-z0-9_']+", text.lower()) if token]
        n_tok = len(toks)
        n_uniq = len(set(toks))
        avg_len = float(np.mean([len(token) for token in toks])) if n_tok > 0 else 0.0
        ttr = float(n_uniq / n_tok) if n_tok > 0 else 0.0

        tok_counts.append(float(n_tok))
        uniq_counts.append(float(n_uniq))
        avg_tok_lens.append(avg_len)
        ttrs.append(ttr)

    return pd.DataFrame(
        {
            "token_count": np.asarray(tok_counts, dtype=float),
            "unique_token_count": np.asarray(uniq_counts, dtype=float),
            "avg_token_len": np.asarray(avg_tok_lens, dtype=float),
            "lexical_ttr": np.asarray(ttrs, dtype=float),
        }
    )


def safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    """Create stable quantile bins even when the raw series contains many duplicates."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    ranked = numeric.rank(method="first")
    bins = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    return bins.map(lambda x: f"{prefix}_q{int(x) + 1}" if pd.notna(x) else np.nan)
