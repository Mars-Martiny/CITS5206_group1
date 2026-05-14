"""Leaderboard and run-summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_leaderboard(
    sort_by: str = "val_f1_macro",
    ascending: bool = False,
    model_type_filter: str | None = None,
    architecture_filter: str | None = None,
) -> pd.DataFrame:
    """Load ``leaderboard/leaderboard.csv`` with optional filters and sorting."""
    csv_path = _repo_root() / "leaderboard" / "leaderboard.csv"
    df = pd.read_csv(csv_path)

    if model_type_filter is not None:
        mt = model_type_filter.strip().lower()
        if mt not in {"energy", "damage"}:
            raise ValueError("model_type_filter must be 'energy' or 'damage'")
        want_energy = mt == "energy"

        def _flag(v):
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            return s in {"true", "1", "yes"}

        df = df[df["energy_model"].map(_flag) == want_energy]

    if architecture_filter:
        df = df[_matches_architecture_filter(df["model_type"], architecture_filter)]

    df = df.sort_values(by=[sort_by], ascending=ascending, na_position="last")
    return df.reset_index(drop=True)


def _matches_architecture_filter(series: pd.Series, architecture_filter: str) -> pd.Series:
    needle = architecture_filter.strip().lower()
    s_norm = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if needle == "tf_idf":
        return s_norm.eq("tf_idf")
    if needle == "bert":
        return s_norm.eq("bert")
    if needle == "looped_transformer":
        return s_norm.eq("looped_transformer")
    if needle == "bigru":
        return s_norm.eq("bigru") | s_norm.str.startswith("bigru_")
    return s_norm.eq(needle)


def get_model_details(model_dir: str) -> dict[str, Any]:
    """Return the decoded JSON payload for ``*_run_summary.json`` inside ``model_dir``."""
    root = Path(model_dir)
    matches = sorted(root.glob("*_run_summary.json"))
    if not matches:
        raise FileNotFoundError(f"No *_run_summary.json under {root}")
    with open(matches[0], encoding="utf-8") as f:
        return json.load(f)
