"""Dismissible instruction banner component, shared across all pages."""

from __future__ import annotations

import streamlit as st

_INSTRUCTIONS: dict[str, str] = {
    "train": (
        "**Train Model** — Select a model architecture and type, upload or specify your dataset CSV "
        "files, configure hyperparameters, then click **Train Model** to start a run. "
        "Trained models are saved to `trained_models/` and appear automatically in Inference and Metrics."
    ),
    "inference": (
        "**Inference** — Pick trained energy and/or damage models from the dropdown, upload an input "
        "CSV, then click **Run Inference** to score every row. Results show tier counts and a "
        "fatal-flag summary and are downloadable as CSV."
    ),
    "metrics": (
        "**Metrics & Leaderboard** — Browse and compare all trained models. Filter by type or "
        "architecture, sort by any metric, and inspect a specific model's full training summary and "
        "diagnostic plots."
    ),
    "human_review": (
        "**Human Review** — Load inference results to start the three-tier review workflow. "
        "Fatal-flagged rows require mandatory review; HIGH-confidence rows are spot-checked at the "
        "sampling rate configured in Settings; MEDIUM and LOW rows receive progressively fuller review. "
        "Save progress at any time and resume by re-uploading the saved CSV."
    ),
}


def show_instruction_banner(page_key: str) -> None:
    """Render the instruction banner for *page_key*.

    Behaviour depends on the 'Always show instructions' setting:
    - ON  (default): banner is always visible and cannot be dismissed.
    - OFF: an ✕ button lets the user dismiss the banner for the current session.
    """
    text = _INSTRUCTIONS.get(page_key, "")
    if not text:
        return

    always_show: bool = st.session_state.get("settings_always_show_instructions", True)
    dismissed_key = f"banner_dismissed_{page_key}"

    if always_show:
        st.info(text, icon="ℹ️")
        return

    if st.session_state.get(dismissed_key, False):
        return

    col_msg, col_btn = st.columns([22, 1])
    with col_msg:
        st.info(text, icon="ℹ️")
    with col_btn:
        st.write("")
        if st.button("✕", key=f"_close_banner_{page_key}", help="Dismiss"):
            st.session_state[dismissed_key] = True
            st.rerun()
