"""Metrics page — leaderboard and per-model detail view."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api  # noqa: E402

from app.app import list_trained_models
from app.components.banner import show_instruction_banner

_ARCH_DISPLAY_TO_API = {
    "All": None,
    "TF-IDF": "tf_idf",
    "BiGRU": "bigru",
    "BERT": "bert",
    "Looped Transformer": "looped_transformer",
}

_MODEL_TYPE_DISPLAY_TO_API = {
    "All": None,
    "Energy": "energy",
    "Damage": "damage",
}

_LEADERBOARD_COLS = [
    "timestamp",
    "model_type",
    "energy_model",
    "val_f1_macro",
    "test_f1_macro",
    "val_accuracy",
    "training_time_sec",
    "model_path",
]

_SORT_OPTIONS = [
    "val_f1_macro",
    "test_f1_macro",
    "val_accuracy",
    "val_f1_weighted",
    "training_time_sec",
]

st.title("Metrics & Leaderboard")
show_instruction_banner("metrics")

filter_cols = st.columns(5)
with filter_cols[0]:
    model_type_display = st.selectbox("Model Type", list(_MODEL_TYPE_DISPLAY_TO_API))
with filter_cols[1]:
    arch_display = st.selectbox("Architecture", list(_ARCH_DISPLAY_TO_API))
with filter_cols[2]:
    sort_by = st.selectbox("Sort by", _SORT_OPTIONS)
with filter_cols[3]:
    ascending = st.checkbox("Ascending", value=False)
with filter_cols[4]:
    top_n = st.number_input("Show top N", min_value=5, max_value=200, value=30)

model_type_filter = _MODEL_TYPE_DISPLAY_TO_API[model_type_display]
arch_filter = _ARCH_DISPLAY_TO_API[arch_display]

try:
    df = api.get_leaderboard(
        sort_by=sort_by,
        ascending=ascending,
        model_type_filter=model_type_filter,
        architecture_filter=arch_filter,
    )
    present_cols = [c for c in _LEADERBOARD_COLS if c in df.columns]
    st.dataframe(df[present_cols].head(int(top_n)), use_container_width=True)
except FileNotFoundError:
    st.warning("No leaderboard.csv found yet. Train at least one model first.")
    df = None
except Exception as exc:
    st.error(f"Failed to load leaderboard: {exc}")
    df = None

st.subheader("Inspect a Model")

entries = list_trained_models()
model_labels = [""] + [e.label for e in entries]
model_paths = {e.label: str(e.path) for e in entries}

inspect_col_a, inspect_col_b = st.columns([2, 1])
with inspect_col_a:
    manual_path = st.text_input("Model directory path (paste from table or type)")
with inspect_col_b:
    selected_label = st.selectbox("Or pick from list", model_labels)

model_dir_to_load = manual_path.strip() if manual_path.strip() else model_paths.get(selected_label, "")

if st.button("Load Details") and model_dir_to_load:
    try:
        details = api.get_model_details(model_dir_to_load)
        st.json(details)

        plots = list(Path(model_dir_to_load).glob("*_combined_metrics_plot.png")) + list(
            Path(model_dir_to_load).glob("*_test_confusion_matrix.png")
        )
        if plots:
            with st.expander("Training plots", expanded=True):
                for plot_path in plots:
                    st.image(str(plot_path), caption=plot_path.name, use_container_width=True)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Failed to load model details: {exc}")
