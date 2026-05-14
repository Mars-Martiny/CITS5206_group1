"""Settings page — application-wide configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.banner import show_instruction_banner

st.title("Settings")

show_instruction_banner("settings")

st.subheader("Instructions")

always_show: bool = st.checkbox(
    "Always show instructions",
    value=st.session_state.get("settings_always_show_instructions", True),
    help=(
        "When enabled, the instruction banner is always visible on every page. "
        "Disable this to allow each banner to be dismissed individually."
    ),
)
st.session_state["settings_always_show_instructions"] = always_show

st.divider()

st.subheader("Human Review — HIGH Confidence Sampling")

high_sample_pct: int = int(
    st.number_input(
        "Sampling percentage for HIGH auto-classified rows",
        min_value=1,
        max_value=100,
        value=st.session_state.get("settings_high_sample_pct", 10),
        step=1,
        format="%d",
        help=(
            "Whole number percentage of HIGH-confidence predictions included in the "
            "spot-check review queue. Enter 10 for 10%, 25 for 25%, etc."
        ),
    )
)
st.session_state["settings_high_sample_pct"] = high_sample_pct
st.caption(f"Currently sampling **{high_sample_pct}%** of HIGH-confidence predictions for human review.")
