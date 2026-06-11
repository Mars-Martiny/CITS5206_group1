"""Retraining page — continue or refresh an existing trained model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app import list_trained_models, save_uploaded_file

try:
    import torch
except Exception:  # pragma: no cover - UI can still render without torch import.
    torch = None


_SORT_OPTIONS = [
    "val_f1_macro",
    "test_f1_macro",
    "val_accuracy",
    "test_accuracy",
    "training_time_sec",
]

_TASK_FILTERS = {
    "Energy Type": "energy",
    "Damage Potential": "damage",
    "All saved models": None,
}

_RETRAIN_MODES = {
    "Auto — continue BERT, refresh TF-IDF/default": "auto",
    "Continue — keep compatible saved artifacts/checkpoint": "continue",
    "Refresh — rebuild artifacts and train a new version": "refresh",
}

_BEST_METRICS = [
    "f1_macro",
    "f1_weighted",
    "accuracy",
    "loss",
    "precision_macro",
    "recall_macro",
    "precision_weighted",
    "recall_weighted",
]

_NONE_LABEL = "— none —"


def _optional_int(value: Any) -> int | None:
    """Return a positive int or None."""
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """Return a float or None."""
    try:
        parsed = float(value)
        return parsed
    except (TypeError, ValueError):
        return None


def _load_summary(model_dir: str | Path) -> dict[str, Any] | None:
    """Load the first run-summary JSON under a saved-model directory."""
    root = Path(model_dir)
    summaries = sorted(root.glob("*_run_summary.json"))
    if not summaries:
        return None

    try:
        with open(summaries[0], encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sort_key(entry, sort_by: str) -> float:
    """Sort saved models using the same summary fields as inference_page.py."""
    summary = _load_summary(entry.path)
    if not summary:
        return 0.0

    try:
        if sort_by in summary and not isinstance(summary[sort_by], (dict, list)):
            value = summary.get(sort_by)
            return float(value) if value is not None else 0.0

        if sort_by.startswith("val_"):
            metric = sort_by[4:]
            values = (
                summary.get("history", {})
                .get("training", {})
                .get("val", {})
                .get(metric, [])
            )
            return max((float(v) for v in values if v is not None), default=0.0)

        if sort_by.startswith("test_"):
            metric = sort_by[5:]
            value = summary.get("history", {}).get("test", {}).get(metric)
            if isinstance(value, list):
                return max((float(v) for v in value if v is not None), default=0.0)
            return float(value) if value is not None else 0.0

    except Exception:
        return 0.0

    return 0.0


def _build_model_selectbox(
    label: str,
    task_filter: str | None,
    sort_by: str,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Build a saved-model dropdown and return selected label/path/summary."""
    entries = list_trained_models(task_filter=task_filter)
    entries.sort(key=lambda e: _sort_key(e, sort_by), reverse=True)

    options = [_NONE_LABEL] + [entry.label for entry in entries]
    paths = {entry.label: str(entry.path) for entry in entries}

    selected = st.selectbox(label, options)
    if selected == _NONE_LABEL:
        return selected, None, None

    model_dir = paths[selected]
    summary = _load_summary(model_dir)

    if summary:
        metric_name = summary.get("best_metric_name", "metric")
        metric_value = summary.get("best_metric_value", 0.0)
        try:
            st.caption(f"Best {metric_name}: {float(metric_value):.4f}")
        except Exception:
            st.caption(f"Best {metric_name}: {metric_value}")

    return selected, model_dir, summary


def _normalise_model_family(summary: dict[str, Any] | None) -> str | None:
    """Infer architecture family from saved summary config."""
    if not summary:
        return None

    config = summary.get("config", {})
    model_type = str(config.get("model_type", "")).strip().lower()

    if model_type in {"tfidf", "tf-idf", "tf_idf"}:
        return "tf_idf"
    if model_type == "bert" or model_type.startswith("safetybert"):
        return "bert"
    if model_type.startswith("bigru"):
        return "bigru"
    if model_type == "looped_transformer":
        return "looped_transformer"

    return model_type or None


def _default_text_col(summary: dict[str, Any] | None) -> str:
    """Infer text column from summary config/parameters."""
    if not summary:
        return "description"

    config = summary.get("config", {})
    parameters = config.get("parameters", {})

    return (
        config.get("text_col")
        or parameters.get("text_col")
        or "description"
    )


def _default_run_name(summary: dict[str, Any] | None) -> str:
    """Generate a sensible retrain run name from the old model summary."""
    if not summary:
        return ""

    config = summary.get("config", {})
    base_name = config.get("run_name") or config.get("save_name") or "model"
    return f"{base_name}_retrain"


def _safe_for_json(value: Any, *, _depth: int = 0) -> Any:
    """Convert common training-result objects into Streamlit JSON-safe values."""
    if _depth > 8:
        return repr(value)

    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    if isinstance(value, Path):
        return str(value)

    if torch is not None and isinstance(value, torch.Tensor):
        return f"<Tensor shape={tuple(value.shape)} dtype={value.dtype}>"

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, val in value.items():
            key_str = str(key)
            if key_str in {
                "best_model_state_dict",
                "model",
                "train_dl",
                "valid_dl",
                "test_dl",
                "optimiser",
                "scheduler",
                "criterion",
            }:
                output[key_str] = f"<omitted {type(val).__name__}>"
            else:
                output[key_str] = _safe_for_json(val, _depth=_depth + 1)
        return output

    if isinstance(value, (list, tuple)):
        if len(value) > 50:
            return [
                _safe_for_json(v, _depth=_depth + 1)
                for v in value[:50]
            ] + [f"<{len(value) - 50} more items omitted>"]
        return [_safe_for_json(v, _depth=_depth + 1) for v in value]

    return repr(value)


def _show_selected_model_details(summary: dict[str, Any] | None, model_dir: str) -> None:
    """Render compact metadata for the selected base model."""
    if not summary:
        st.warning("No readable run summary was found for this model directory.")
        st.code(model_dir)
        return

    config = summary.get("config", {})
    history = summary.get("history", {})
    test_metrics = history.get("test", {})

    model_type = config.get("model_type", "unknown")
    task = "Energy Type" if bool(config.get("energy_model", False)) else "Damage Potential"
    best_metric_name = summary.get("best_metric_name", "metric")
    best_metric_value = summary.get("best_metric_value", "—")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Architecture", str(model_type))
    with cols[1]:
        st.metric("Task", task)
    with cols[2]:
        try:
            st.metric(f"Best {best_metric_name}", f"{float(best_metric_value):.4f}")
        except Exception:
            st.metric(f"Best {best_metric_name}", str(best_metric_value))
    with cols[3]:
        test_f1 = test_metrics.get("f1_macro", ["—"])
        if isinstance(test_f1, list):
            test_f1 = test_f1[-1] if test_f1 else "—"
        try:
            st.metric("Test F1 macro", f"{float(test_f1):.4f}")
        except Exception:
            st.metric("Test F1 macro", str(test_f1))

    st.caption(f"Base model directory: `{model_dir}`")


def _resolve_dataset_paths(use_upload: bool):
    """Render dataset controls and return path/upload objects."""
    if use_upload:
        train_file = st.file_uploader("Train CSV", type=["csv"], key="retrain_train_upload")
        valid_file = st.file_uploader("Validation CSV", type=["csv"], key="retrain_valid_upload")
        test_file = st.file_uploader("Test CSV", type=["csv"], key="retrain_test_upload")
        return train_file, valid_file, test_file, None, None, None

    train_path_str = st.text_input("Train CSV path")
    valid_path_str = st.text_input("Validation CSV path")
    test_path_str = st.text_input("Test CSV path")
    return None, None, None, train_path_str, valid_path_str, test_path_str


st.title("Retraining")
st.info(
    "**Retraining** — Select an existing saved model, provide updated train/validation/test "
    "CSV splits, choose whether to continue from compatible artifacts or refresh the artifact "
    "pipeline, then launch a new versioned run. The original model directory is not overwritten.",
    icon="ℹ️",
)

st.subheader("Base model")

selection_cols = st.columns(2)
with selection_cols[0]:
    task_display = st.radio("Model Type", list(_TASK_FILTERS), horizontal=True)
with selection_cols[1]:
    sort_by = st.selectbox("Sort models by", _SORT_OPTIONS)

task_filter = _TASK_FILTERS[task_display]
_, selected_model_dir, selected_summary = _build_model_selectbox(
    "Saved model",
    task_filter=task_filter,
    sort_by=sort_by,
)

with st.expander("Manual model directory override"):
    manual_model_dir = st.text_input(
        "Model directory",
        value="",
        placeholder="trained_models/20260516_120000_model_run",
        help="Optional. Use this when the model is not shown in the dropdown.",
    )

if manual_model_dir.strip():
    selected_model_dir = manual_model_dir.strip()
    selected_summary = _load_summary(selected_model_dir)

if selected_model_dir:
    _show_selected_model_details(selected_summary, selected_model_dir)
else:
    st.warning("Select a saved model or enter a manual model directory.")

model_family = _normalise_model_family(selected_summary)

st.subheader("Retraining data")

use_upload = st.toggle("Upload files", value=True)
train_file, valid_file, test_file, train_path_str, valid_path_str, test_path_str = (
    _resolve_dataset_paths(use_upload)
)

text_col = st.text_input("Text column", value=_default_text_col(selected_summary))

st.subheader("Retraining setup")

mode_display = st.radio(
    "Retraining mode",
    list(_RETRAIN_MODES),
    horizontal=False,
    help=(
        "Auto generally continues BERT-like models from the checkpoint and refreshes "
        "TF-IDF models so vectorizers/embedding artifacts are rebuilt safely."
    ),
)
mode = _RETRAIN_MODES[mode_display]

run_name_default = _default_run_name(selected_summary)
run_name = st.text_input(
    "Run name",
    value=run_name_default,
    help="Leave blank to let the retrain API choose a name.",
)

parent_dir = st.text_input(
    "Output parent directory",
    value="trained_models",
    help="New retrained model versions are saved under this parent directory.",
)

st.subheader("Hyperparameters")

hp_cols = st.columns(4)
with hp_cols[0]:
    epochs = st.number_input("epochs", min_value=1, value=None, placeholder="default")
with hp_cols[1]:
    patience = st.number_input("patience", min_value=1, value=None, placeholder="default")
with hp_cols[2]:
    batch_size = st.number_input("batch_size", min_value=1, value=None, placeholder="default")
with hp_cols[3]:
    learning_rate = st.number_input(
        "learning_rate",
        min_value=0.0,
        value=None,
        placeholder="default",
        format="%.8f",
    )

advanced_open = model_family in {"tf_idf", "bert", "bigru", "looped_transformer"}
with st.expander("Advanced retraining options", expanded=advanced_open):
    metric_cols = st.columns(3)
    with metric_cols[0]:
        best_metric = st.selectbox("best_metric", ["default"] + _BEST_METRICS)
    with metric_cols[1]:
        threshold = st.number_input(
            "threshold",
            min_value=0.0,
            max_value=1.0,
            value=None,
            placeholder="default",
            format="%.4f",
        )
    with metric_cols[2]:
        log_leaderboard = st.checkbox("log_leaderboard", value=True)

    hidden_dim = None
    feature_representation = None
    embedding_model_name = None
    fine_tune = None
    pooling = None
    embedding_type = None
    num_loops = None
    d_model = None

    if model_family in {"tf_idf", "bigru"}:
        hidden_dim = st.number_input(
            "hidden_dim",
            min_value=1,
            value=None,
            placeholder="default",
        )

    if model_family == "tf_idf":
        tfidf_cols = st.columns(2)
        with tfidf_cols[0]:
            feature_representation = st.selectbox(
                "feature_representation",
                ["default", "tfidf", "tfidf_embed_avg"],
            )
        with tfidf_cols[1]:
            embedding_model_name = st.text_input(
                "embedding_model_name",
                value="",
                placeholder="default, e.g. adanish91/safetybert",
            )

    if model_family == "bert":
        bert_cols = st.columns(2)
        with bert_cols[0]:
            fine_tune = st.checkbox("fine_tune", value=True)
        with bert_cols[1]:
            pooling = st.selectbox("pooling", ["default", "mean", "cls"])

    if model_family == "bigru":
        embedding_type = st.selectbox(
            "embedding_type",
            ["default", "none", "static", "contextual"],
        )

    if model_family == "looped_transformer":
        loop_cols = st.columns(2)
        with loop_cols[0]:
            num_loops = st.number_input(
                "num_loops",
                min_value=1,
                value=None,
                placeholder="default",
            )
        with loop_cols[1]:
            d_model = st.number_input(
                "d_model",
                min_value=1,
                value=None,
                placeholder="default",
            )

    extra_config_json = st.text_area(
        "Extra train_config JSON",
        value="{}",
        help=(
            "Optional overrides merged into train_config. Use valid JSON, for example "
            '{"weight_decay": 0.01, "use_class_weights": true}.'
        ),
    )

if st.button("Retrain Model", type="primary"):
    errors: list[str] = []

    if not selected_model_dir:
        errors.append("Select a base model or enter a manual model directory.")

    if use_upload:
        if train_file is None:
            errors.append("Train CSV is required.")
        if valid_file is None:
            errors.append("Validation CSV is required.")
        if test_file is None:
            errors.append("Test CSV is required.")
    else:
        if not train_path_str:
            errors.append("Train CSV path is required.")
        if not valid_path_str:
            errors.append("Validation CSV path is required.")
        if not test_path_str:
            errors.append("Test CSV path is required.")

    try:
        extra_cfg = json.loads(extra_config_json or "{}")
        if not isinstance(extra_cfg, dict):
            errors.append("Extra train_config JSON must decode to an object/dictionary.")
    except json.JSONDecodeError as exc:
        extra_cfg = {}
        errors.append(f"Extra train_config JSON is invalid: {exc}")

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    if use_upload:
        train_path = str(save_uploaded_file(train_file))
        valid_path = str(save_uploaded_file(valid_file))
        test_path = str(save_uploaded_file(test_file))
    else:
        train_path = str(train_path_str)
        valid_path = str(valid_path_str)
        test_path = str(test_path_str)

    cfg: dict[str, Any] = {}

    if _optional_int(epochs):
        cfg["epochs"] = int(epochs)
    if _optional_int(patience):
        cfg["patience"] = int(patience)
    if _optional_int(batch_size):
        cfg["batch_size"] = int(batch_size)
    if _optional_float(learning_rate) is not None:
        cfg["learning_rate"] = float(learning_rate)
    if run_name.strip():
        cfg["run_name"] = run_name.strip()
    if parent_dir.strip():
        cfg["parent_dir"] = parent_dir.strip()
    if best_metric != "default":
        cfg["best_metric"] = best_metric
    if _optional_float(threshold) is not None:
        cfg["threshold"] = float(threshold)

    cfg["log_leaderboard"] = bool(log_leaderboard)

    if hidden_dim is not None and _optional_int(hidden_dim):
        cfg["hidden_dim"] = int(hidden_dim)
    if feature_representation and feature_representation != "default":
        cfg["feature_representation"] = feature_representation
    if embedding_model_name and embedding_model_name.strip():
        cfg["embedding_model_name"] = embedding_model_name.strip()
    if fine_tune is not None:
        cfg["fine_tune"] = bool(fine_tune)
    if pooling and pooling != "default":
        cfg["pooling"] = pooling
    if embedding_type and embedding_type != "default":
        cfg["embedding_type"] = embedding_type
    if num_loops is not None and _optional_int(num_loops):
        cfg["num_loops"] = int(num_loops)
    if d_model is not None and _optional_int(d_model):
        cfg["d_model"] = int(d_model)

    cfg.update(extra_cfg)

    try:
        import api
        retrain = api.retrain
    except Exception:
        from api.retrain import retrain

    with st.spinner("Retraining in progress…"):
        try:
            result = retrain(
                model_dir=str(selected_model_dir),
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                train_config=cfg or None,
                text_col=text_col,
                mode=mode,
            )
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")
            st.stop()

    result_config = result.get("config", {})
    save_dir = result_config.get("save_dir", "—")
    metric_name = result.get("best_metric_name", "metric")
    metric_value = result.get("best_metric_value", 0.0)

    st.success(
        f"Retraining complete!\n\n"
        f"**Saved to:** `{save_dir}`\n\n"
        f"**Best {metric_name}:** {float(metric_value):.4f}"
        if isinstance(metric_value, (int, float))
        else f"Retraining complete!\n\n**Saved to:** `{save_dir}`\n\n**Best {metric_name}:** {metric_value}"
    )

    test_metrics = result.get("history", {}).get("test", {})
    if test_metrics:
        metric_cols = st.columns(4)
        display_metrics = [
            ("accuracy", "Test accuracy"),
            ("f1_macro", "Test F1 macro"),
            ("f1_weighted", "Test F1 weighted"),
            ("auto_classification_rate", "Auto-classification"),
        ]

        for idx, (key, label) in enumerate(display_metrics):
            value = test_metrics.get(key)
            if isinstance(value, list):
                value = value[-1] if value else None
            if value is None:
                continue
            with metric_cols[idx % 4]:
                try:
                    st.metric(label, f"{float(value):.4f}")
                except Exception:
                    st.metric(label, str(value))
