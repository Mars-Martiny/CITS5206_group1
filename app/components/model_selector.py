"""Reusable model-picker widget."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app import list_trained_models

_NONE_LABEL = "— none —"


def model_selector_widget(
    label: str,
    task_filter: str | None = None,
    key: str = "model_selector",
) -> str | None:
    """Render a selectbox of trained models and return the selected path or None."""
    entries = list_trained_models(task_filter=task_filter)
    options = [_NONE_LABEL] + [e.label for e in entries]
    paths = {e.label: str(e.path) for e in entries}

    selected = st.selectbox(label, options, key=key)
    if selected == _NONE_LABEL:
        return None
    return paths.get(selected)
