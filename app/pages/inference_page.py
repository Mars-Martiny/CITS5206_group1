"""Inference page — run a trained model on new incident data."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app import list_trained_models, save_uploaded_file
from app.components.banner import show_instruction_banner

_SORT_OPTIONS = ["val_f1_macro", "test_f1_macro", "val_accuracy", "training_time_sec"]
_NONE_LABEL = "— none —"


def _build_model_selectbox(label: str, task_filter: str, sort_by: str) -> tuple[str, str | None]:
    entries = list_trained_models(task_filter=task_filter)
    entries.sort(key=lambda e: _sort_key(e, sort_by))
    options = [_NONE_LABEL] + [e.label for e in entries]
    paths = {e.label: str(e.path) for e in entries}

    selected = st.selectbox(label, options)
    if selected == _NONE_LABEL:
        return _NONE_LABEL, None

    path = paths[selected]
    import json

    summaries = sorted(Path(path).glob("*_run_summary.json"))
    if summaries:
        with open(summaries[0], encoding="utf-8") as f:
            s = json.load(f)
        metric_name = s.get("best_metric_name", "metric")
        metric_value = s.get("best_metric_value", 0.0)
        st.caption(f"Best {metric_name}: {metric_value:.4f}")

    return selected, path


def _sort_key(entry, sort_by: str):
    import json

    summaries = sorted(entry.path.glob("*_run_summary.json"))
    if not summaries:
        return 0.0
    try:
        with open(summaries[0], encoding="utf-8") as f:
            s = json.load(f)
        val = s.get(sort_by) or s.get("history", {}).get(sort_by, 0.0)
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


st.title("Inference")
show_instruction_banner("inference")

sort_by = st.selectbox("Sort models by", _SORT_OPTIONS)

st.subheader("Energy Model (optional)")
_, energy_model_dir = _build_model_selectbox("Energy model", "energy", sort_by)

st.subheader("Damage Model (optional)")
_, damage_model_dir = _build_model_selectbox("Damage model", "damage", sort_by)

st.subheader("Dataset")
input_csv = st.file_uploader("Input CSV", type=["csv"])
output_path = st.text_input("Output file path", value="/tmp/inference_output.csv")
text_col = st.text_input("Text column", value="description")

if st.button("Run Inference", type="primary"):
    if energy_model_dir is None and damage_model_dir is None:
        st.error("Select at least one model (energy or damage).")
        st.stop()
    if input_csv is None:
        st.error("Upload an input CSV.")
        st.stop()

    dataset_path = str(save_uploaded_file(input_csv))

    import api

    with st.spinner("Running inference…"):
        try:
            df = api.infer(
                dataset_path=dataset_path,
                energy_model_dir=energy_model_dir,
                damage_model_dir=damage_model_dir,
                output_path=output_path,
                text_col=text_col,
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

    st.success(f"Scored {len(df)} rows. Saved to {output_path}")

    metric_cols = st.columns(4)
    col_idx = 0

    for score_col, model_label in [("energy_score", "Energy"), ("damage_score", "Damage")]:
        if score_col in df.columns:
            for tier in ("HIGH", "MEDIUM", "LOW"):
                count = int((df[score_col] == tier).sum())
                with metric_cols[col_idx % 4]:
                    st.metric(f"{model_label} {tier}", count)
                col_idx += 1

    if "fatal_flag" in df.columns and damage_model_dir is not None:
        fatal_count = int((df["fatal_flag"] == "YES").sum())
        st.markdown(
            f"<span style='color:red;font-weight:bold'>Fatal-flagged: {fatal_count}</span>",
            unsafe_allow_html=True,
        )

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="inference_output.csv",
        mime="text/csv",
    )
