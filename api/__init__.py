"""High-level orchestration helpers for training, exporting, inference, and reporting.

This module intentionally avoids eagerly importing inference/model-loading
dependencies. The Streamlit training page imports `api` before calling
`api.train(...)`; eager imports here can pull in BERT/Transformers modules even
for TF-IDF training.
"""

from typing import Any

from .train import train

__all__ = [
    "train",
    "load_model",
    "infer",
    "get_leaderboard",
    "get_model_details",
    "retrain",
]


def __getattr__(name: str) -> Any:
    """Lazily expose heavier API functions only when requested."""
    if name == "load_model":
        from .loader import load_model

        return load_model

    if name == "infer":
        from .infer import infer

        return infer

    if name == "get_leaderboard":
        from .metrics import get_leaderboard

        return get_leaderboard

    if name == "get_model_details":
        from .metrics import get_model_details

        return get_model_details

    if name == "retrain":
        from .retrain import retrain

        return retrain

    raise AttributeError(f"module 'api' has no attribute {name!r}")