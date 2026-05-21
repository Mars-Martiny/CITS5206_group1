"""Retrain or fine-tune exported model directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from api.loader import load_model
from api.train import _rename_columns, train as train_from_splits

from experiment_setup.bert_runner import bert_continue_train
from experiment_setup.tf_idf_runner import tf_idf_continue_train


Retrainer = Callable[
    [dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, str, dict[str, Any]],
    dict[str, Any],
]


_RETRAINERS: dict[str, Retrainer] = {}


def register_retrainer(model_type: str, fn: Retrainer) -> None:
    """Register a model-specific continue-training implementation."""
    _RETRAINERS[model_type.lower()] = fn


def _read_csv(path: str) -> pd.DataFrame:
    return _rename_columns(pd.read_csv(path))


def _normalise_architecture(model_type: str) -> str:
    mt = str(model_type).strip().lower()

    if mt in {"tfidf", "tf-idf", "tf_idf"}:
        return "tf_idf"

    if mt == "bert" or mt.startswith("safetybert"):
        return "bert"

    if mt.startswith("bigru"):
        return "bigru"

    if mt == "looped_transformer":
        return "looped_transformer"

    return mt


def _infer_target(bundle: dict[str, Any]) -> str:
    return "energy" if bool(bundle.get("energy_model")) else "damage"


def _default_text_col(bundle: dict[str, Any], text_col: str | None) -> str:
    if text_col:
        return text_col

    artifacts = bundle.get("artifacts", {})
    config = bundle.get("config", {})

    return (
        artifacts.get("text_col")
        or config.get("text_col")
        or config.get("parameters", {}).get("text_col")
        or "description"
    )


def _merge_refresh_config(bundle: dict[str, Any], train_config: dict | None) -> dict:
    """Build a config for full artifact-refresh retraining.

    This deliberately keeps the implementation conservative: use saved artifact
    hyperparameters as defaults, then let caller overrides win.
    """
    artifacts = bundle.get("artifacts", {})
    old_config = bundle.get("config", {})

    cfg = {
        "save": True,
        "run_name": f"{old_config.get('run_name', old_config.get('save_name', 'model'))}_retrain",
        "threshold": old_config.get("threshold", 0.8),
        "temperature": old_config.get("temperature", 1.0),
        "use_temperature": old_config.get("use_temperature", False),
        "batch_size": artifacts.get("batch_size"),
    }

    # Preserve common model-specific artifact settings when available.
    for key in (
        "hidden_dim",
        "feature_representation",
        "embedding_model_name",
        "fine_tune",
        "pooling",
        "dropout",
        "max_length",
        "weight_decay",
    ):
        if key in artifacts:
            cfg[key] = artifacts[key]

    # Remove None defaults so existing runner defaults still apply.
    cfg = {k: v for k, v in cfg.items() if v is not None}
    cfg.update(train_config or {})
    return cfg


def retrain(
    model_dir: str,
    train_path: str,
    valid_path: str,
    test_path: str,
    train_config: dict | None = None,
    text_col: str | None = None,
    mode: str = "auto",
) -> dict[str, Any]:
    """Retrain or fine-tune a saved model directory.

    Args:
        model_dir: Existing trained model directory containing model, summary,
            and artifact files.
        train_path: CSV training split.
        valid_path: CSV validation split.
        test_path: CSV test split.
        train_config: Optional training overrides.
        text_col: Text column. If omitted, uses saved artifact/config value.
        mode:
            "auto": BERT continues from saved weights; TF-IDF refreshes artifacts.
            "continue": Continue training from saved checkpoint where supported.
            "refresh": Rebuild artifacts and train with the existing runner.

    Returns:
        New run summary from the shared training loop.
    """
    bundle = load_model(model_dir)
    architecture = _normalise_architecture(bundle["model_type"])
    target = _infer_target(bundle)
    text_col = _default_text_col(bundle, text_col)

    mode = mode.lower().strip()
    if mode not in {"auto", "continue", "refresh"}:
        raise ValueError("mode must be one of: 'auto', 'continue', 'refresh'")

    if mode == "auto":
        mode = "continue" if architecture == "bert" else "refresh"

    if mode == "refresh":
        cfg = _merge_refresh_config(bundle, train_config)
        return train_from_splits(
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            model_type=target,
            architecture=architecture,
            train_config=cfg,
            text_col=text_col,
        )

    train_df = _read_csv(train_path)
    valid_df = _read_csv(valid_path)
    test_df = _read_csv(test_path)

    retrainer = _RETRAINERS.get(architecture)
    if retrainer is None:
        raise NotImplementedError(
            f"Continue-mode retraining is not implemented for {architecture!r}. "
            "Use mode='refresh', or register a retrainer with register_retrainer(...)."
        )

    cfg = dict(train_config or {})
    return retrainer(bundle, train_df, valid_df, test_df, text_col, cfg)


# Register built-in retrainers for supported architectures.
# extend with other architectures as needed, ensuring that the retrainer function is implemented and imported correctly in this module.
register_retrainer("bert", bert_continue_train)
register_retrainer("tf_idf", tf_idf_continue_train)

