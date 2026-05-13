"""Streamlit entry point — multi-page app with shared helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import NamedTuple

import streamlit as st

TRAINED_MODELS_DIR = Path(__file__).resolve().parents[1] / "trained_models"
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


class ModelEntry(NamedTuple):
    """Lightweight descriptor for a saved model run."""

    label: str
    path: Path


def list_trained_models(task_filter: str | None = None) -> list[ModelEntry]:
    """Scan TRAINED_MODELS_DIR and return labelled model entries.

    task_filter: "energy" or "damage" — matched against run_summary config.energy_model.
    """
    entries: list[ModelEntry] = []

    if not TRAINED_MODELS_DIR.exists():
        return entries

    for model_dir in sorted(TRAINED_MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        summaries = sorted(model_dir.glob("*_run_summary.json"))
        if not summaries:
            continue

        try:
            with open(summaries[0], encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue

        if task_filter is not None:
            is_energy = bool(summary.get("config", {}).get("energy_model", False))
            if task_filter == "energy" and not is_energy:
                continue
            if task_filter == "damage" and is_energy:
                continue

        timestamp = summary.get("config", {}).get("timestamp", model_dir.name)
        model_type = summary.get("config", {}).get("model_type", "unknown")
        metric_name = summary.get("best_metric_name", "metric")
        metric_value = summary.get("best_metric_value", 0.0)

        label = f"{timestamp} — {model_type} [{metric_name}={metric_value:.4f}]"
        entries.append(ModelEntry(label=label, path=model_dir))

    return entries


def save_uploaded_file(uploaded_file) -> Path:
    """Persist a Streamlit UploadedFile to a temp path and return it."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as f:
        f.write(uploaded_file.getbuffer())
        return Path(f.name)


DOCS_INDEX = Path(__file__).resolve().parents[1] / "docs" / "build" / "html" / "index.html"


def main():
    """Run the Streamlit multi-page app."""
    st.set_page_config(page_title="Incident Classifier", layout="wide")

    train_pg = st.Page("pages/train_page.py", title="Train", icon="🏋️")
    infer_pg = st.Page("pages/inference_page.py", title="Inference", icon="🔍")
    metrics_pg = st.Page("pages/metrics_page.py", title="Metrics", icon="📊")
    review_pg = st.Page("pages/human_review_page.py", title="Human Review", icon="👁️")
    report_pg = st.Page("pages/report_page.py", title="Report", icon="📄")
    settings_pg = st.Page("pages/settings_page.py", title="Settings", icon="⚙️")

    pg = st.navigation([train_pg, infer_pg, metrics_pg, review_pg, report_pg, settings_pg])

    with st.sidebar:
        if DOCS_INDEX.exists():
            docs_url = DOCS_INDEX.as_uri()
        else:
            docs_url = None
        if docs_url:
            st.markdown(f"[📚 Documentation]({docs_url})", unsafe_allow_html=True)
        else:
            st.caption("Documentation not built yet. Run `make -C docs html`.")

    pg.run()


if __name__ == "__main__":
    main()
