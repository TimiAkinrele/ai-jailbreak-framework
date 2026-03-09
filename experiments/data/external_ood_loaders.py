"""Compatibility wrapper for external OOD loaders.

Canonical implementation lives in src.data.external_ood_loaders.
"""

from src.data.external_ood_loaders import (  # noqa: F401
    DEFAULT_QUALIFIRE_LOCAL_CSV,
    first_available_split,
    load_notinject_benign,
    load_qualifire_local,
    load_qualifire_jailbreak_attacks,
    load_qualifire_standard,
)

__all__ = [
    "DEFAULT_QUALIFIRE_LOCAL_CSV",
    "first_available_split",
    "load_notinject_benign",
    "load_qualifire_local",
    "load_qualifire_jailbreak_attacks",
    "load_qualifire_standard",
]
