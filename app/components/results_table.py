"""Styled DataFrame display component."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def results_table(df: pd.DataFrame, highlight_col: str | None = None) -> None:
    """Display a DataFrame with optional tier-based row highlighting."""
    if highlight_col and highlight_col in df.columns:

        def _row_color(row):
            tier = row.get(highlight_col, "")
            if tier == "HIGH":
                return ["background-color: #ffd6d6"] * len(row)
            if tier == "MEDIUM":
                return ["background-color: #fff3cd"] * len(row)
            if tier == "LOW":
                return ["background-color: #d6eaff"] * len(row)
            return [""] * len(row)

        styled = df.style.apply(_row_color, axis=1)
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
