"""Train page — configure and launch a training run."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app import save_uploaded_file
from app.components.banner import show_instruction_banner

_ARCH_MAP = {
    "TF-IDF": "tf_idf",
    "BiGRU": "bigru",
    "BERT": "bert",
    "Looped Transformer": "looped_transformer",
}

_MODEL_TYPE_MAP = {
    "Energy Type": "energy",
    "Damage Potential": "damage",
}


def _optional_int(val) -> int | None:
    try:
        v = int(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


st.title("Train Model")
show_instruction_banner("train")

col_mt, col_arch = st.columns(2)
with col_mt:
    model_type_display = st.radio("Model Type", list(_MODEL_TYPE_MAP))
with col_arch:
    arch_display = st.selectbox("Architecture", list(_ARCH_MAP))

arch_key = _ARCH_MAP[arch_display]

st.subheader("Dataset")
use_upload = st.toggle("Upload files", value=True)

if use_upload:
    train_file = st.file_uploader("Train CSV", type=["csv"], key="train_upload")
    valid_file = st.file_uploader("Validation CSV", type=["csv"], key="valid_upload")
    test_file = st.file_uploader("Test CSV", type=["csv"], key="test_upload")
    train_path_str = valid_path_str = test_path_str = None
else:
    train_file = valid_file = test_file = None
    train_path_str = st.text_input("Train CSV path")
    valid_path_str = st.text_input("Validation CSV path")
    test_path_str = st.text_input("Test CSV path")

text_col = st.text_input("Text column", value="description")

st.subheader("Hyperparameters")

hp_cols = st.columns(3)
with hp_cols[0]:
    epochs = st.number_input("epochs", min_value=1, value=None, placeholder="default")
with hp_cols[1]:
    patience = st.number_input("patience", min_value=1, value=None, placeholder="default")

if arch_key in ("tf_idf", "bigru"):
    with hp_cols[2]:
        hidden_dim = st.number_input("hidden_dim", min_value=1, value=None, placeholder="default")
else:
    hidden_dim = None

if arch_key in ("bigru", "bert", "looped_transformer"):
    batch_size = st.number_input("batch_size", min_value=1, value=None, placeholder="default")
else:
    batch_size = None

embedding_type = fine_tune = pooling = num_loops = d_model = None

if arch_key == "bigru":
    embedding_type = st.selectbox("embedding_type", ["none", "static", "contextual"])

if arch_key == "bert":
    fine_tune = st.checkbox("fine_tune", value=False)
    pooling = st.selectbox("pooling", ["mean", "cls"])

if arch_key == "looped_transformer":
    lt_cols = st.columns(2)
    with lt_cols[0]:
        num_loops = st.number_input("num_loops", min_value=1, value=None, placeholder="default")
    with lt_cols[1]:
        d_model = st.number_input("d_model", min_value=1, value=None, placeholder="default")

if st.button("Train Model", type="primary"):
    errors = []
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

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    if use_upload:
        train_path = str(save_uploaded_file(train_file))
        valid_path = str(save_uploaded_file(valid_file))
        test_path = str(save_uploaded_file(test_file))
    else:
        train_path, valid_path, test_path = train_path_str, valid_path_str, test_path_str

    cfg: dict = {}
    if _optional_int(epochs):
        cfg["epochs"] = int(epochs)
    if _optional_int(patience):
        cfg["patience"] = int(patience)
    if hidden_dim and _optional_int(hidden_dim):
        cfg["hidden_dim"] = int(hidden_dim)
    if batch_size and _optional_int(batch_size):
        cfg["batch_size"] = int(batch_size)
    if embedding_type and embedding_type != "none":
        cfg["embedding_type"] = embedding_type
    if fine_tune is not None:
        cfg["fine_tune"] = fine_tune
    if pooling:
        cfg["pooling"] = pooling
    if num_loops and _optional_int(num_loops):
        cfg["num_loops"] = int(num_loops)
    if d_model and _optional_int(d_model):
        cfg["d_model"] = int(d_model)

    import api

    with st.spinner("Training in progress…"):
        try:
            result = api.train(
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                model_type=_MODEL_TYPE_MAP[model_type_display],
                architecture=arch_key,
                train_config=cfg or None,
                text_col=text_col,
            )
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

    save_dir = result.get("config", {}).get("save_dir", "—")
    metric_name = result.get("best_metric_name", "metric")
    metric_value = result.get("best_metric_value", 0.0)

    st.success(
        f"Training complete!\n\n"
        f"**Saved to:** `{save_dir}`\n\n"
        f"**Best {metric_name}:** {metric_value:.4f}"
    )

    with st.expander("Full training summary"):
        st.json(result)
