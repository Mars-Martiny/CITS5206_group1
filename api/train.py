"""Unified training entry point for CSV splits across architectures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_COLUMN_MAP_PATH = Path(__file__).resolve().parents[1] / "column_map.json"


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    with open(_COLUMN_MAP_PATH, encoding="utf-8") as f:
        col_map: dict[str, str] = json.load(f)
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def train(
    train_path: str,
    valid_path: str,
    test_path: str,
    model_type: str,
    architecture: str,
    train_config: dict | None = None,
    text_col: str = "description",
) -> dict[str, Any]:
    """Train a classifier on labelled CSV splits and return the training summary.

    Args:
        train_path: Path to the training CSV.
        valid_path: Path to the validation CSV.
        test_path: Path to the test CSV.
        model_type: ``"energy"`` or ``"damage"`` controlling the supervised target.
        architecture: One of ``tf_idf``, ``bigru``, ``bert``, ``looped_transformer``.
        train_config: Optional hyperparameters forwarded to the selected runner unchanged.
        text_col: Text column present in each CSV.

    Returns:
        Summary dict produced by ``modules.training_loop.train_model_loop`` (via ``training``).

    Raises:
        ValueError: If ``model_type`` or ``architecture`` is unsupported.
    """
    if model_type not in ("energy", "damage"):
        raise ValueError(f"model_type must be 'energy' or 'damage'; got {model_type!r}")
    energy_model = model_type == "energy"

    train_df = _rename_columns(pd.read_csv(train_path))
    valid_df = _rename_columns(pd.read_csv(valid_path))
    test_df = _rename_columns(pd.read_csv(test_path))

    arch = architecture.strip().lower()
    cfg = train_config or {}

    if arch == "tf_idf":
        from experiment_setup.tf_idf_runner import tf_idf_run_single

        return tf_idf_run_single(
            train_df, valid_df, test_df,
            text_col=text_col,
            energy_model=energy_model,
            train_config=cfg,
        )

    if arch == "bigru":
        from experiment_setup.bi_gru_runner import bigru_run_single

        return bigru_run_single(
            train_df, valid_df, test_df,
            text_col=text_col,
            energy_model=energy_model,
            train_config=cfg,
        )

    if arch == "bert":
        from experiment_setup.bert_runner import bert_run_single

        return bert_run_single(
            train_df, valid_df, test_df,
            text_col=text_col,
            energy_model=energy_model,
            train_config=cfg,
        )

    if arch == "looped_transformer":
        from experiment_setup.looped_transformer_runner import looped_transformer_run_single

        return looped_transformer_run_single(
            train_df, valid_df, test_df,
            text_col=text_col,
            energy_model=energy_model,
            train_config=cfg,
        )

    raise ValueError(
        "architecture must be one of tf_idf, bigru, bert, looped_transformer; "
        f"got {architecture!r}"
    )
