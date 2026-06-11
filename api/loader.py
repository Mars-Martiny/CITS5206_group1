"""Load exported training checkpoints for inference."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch

from implementations.bert_transformer import BertClassifier
from implementations.looped_transformer import LoopedTransformer
from implementations.simple_bi_gru import BiGRUClassifier
from implementations.tf_idf import TFIDFClassifier
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_embedding import BertEmbeddingBackend
from modules.embedding.bert_tokenizer import BertTokenizerWrapper

from ._label_utils import label_encoder_from_class_dict


def _one_glob(model_dir: Path, pattern: str) -> Path:
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        msg = f"No file matching {pattern!r} under {model_dir}"
        raise FileNotFoundError(msg)
    return matches[0]


def load_model(model_dir: str) -> dict[str, Any]:
    """Load a saved model from its directory.

    Args:
        model_dir: Path to ``trained_models/{timestamp}_{name}/``.

    Returns:
        Dict containing the PyTorch ``model``, unpickled ``artifacts``, decoded
        ``label_enc``, and saved ``config`` from ``*_run_summary.json``.
    """
    root = Path(model_dir)
    summary_path = _one_glob(root, "*_run_summary.json")
    artifacts_path = _one_glob(root, "*_artifacts.pkl")
    model_weights_path = _one_glob(root, "*_model.pt")

    with open(summary_path, encoding="utf-8") as sf:
        summary = json.load(sf)

    config = summary["config"]
    model_type = str(config.get("model_type", "")).strip()

    with open(artifacts_path, "rb") as pf:
        artifacts: dict[str, Any] = pickle.load(pf)

    energy_model = bool(artifacts.get("energy_model", config.get("energy_model", False)))
    raw_class_dict = dict(config["class_dict"])
    label_enc = artifacts.get("label_enc") or label_encoder_from_class_dict(raw_class_dict)
    num_classes = int(label_enc.num_classes)

    state_dict = torch.load(model_weights_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(
        model_type=model_type,
        config=config,
        artifacts=artifacts,
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "model_type": model_type,
        "energy_model": energy_model,
        "artifacts": artifacts,
        "label_enc": label_enc,
        "class_dict": {int(k): str(v) for k, v in raw_class_dict.items()},
        "config": config,
        "device": device,
        "num_classes": num_classes,
    }


def _build_model(
    model_type: str,
    config: dict[str, Any],
    artifacts: dict[str, Any],
    state_dict: dict[str, Any],
) -> torch.nn.Module:
    mt_lower = model_type.lower()

    if mt_lower == "tf_idf":
        w0 = state_dict["net.0.weight"]
        hidden_infer = int(w0.shape[0])
        vocab_infer = int(w0.shape[1])
        # Find the last Linear layer by scanning state_dict keys
        last_linear_key = max(
            (k for k in state_dict if k.endswith(".weight") and "net." in k),
            key=lambda k: int(k.split(".")[1]),
        )
        num_classes = int(state_dict[last_linear_key].shape[0])
        vec = artifacts.get("vectorizer")
        inferred_inoput_dim = int(artifacts.get("input_dim", vocab_infer))
        inferred_hidden = int(artifacts.get("hidden_dim", hidden_infer))

        label_enc_art = artifacts.get("label_enc")
        if label_enc_art is not None:
            num_classes = int(label_enc_art.num_classes)

        return TFIDFClassifier(
            vocab_size = inferred_inoput_dim,
            num_classes = num_classes,
            hidden_dim = inferred_hidden,
        )

    if mt_lower.startswith("bigru"):
        embedding_type = str(artifacts.get("embedding_type", "none")).lower()

        hidden_dim = int(artifacts.get("hidden_dim", 128))
        dropout_prob = float(artifacts.get("dropout_prob", 0.3))
        freeze_emb = bool(artifacts.get("freeze_emb", False))
        emb_dim = int(artifacts.get("embedding_dim", 128))
        vocab_size = artifacts["vocab_enc"].vocab_size if artifacts.get("vocab_enc") else 1

        energy_model = bool(artifacts.get("energy_model", True))
        le = artifacts["energy_enc"] if energy_model else artifacts["damage_enc"]
        num_classes = int(le.num_classes)

        emb_table_arg = None
        if embedding_type == "none":
            return BiGRUClassifier(
                vocab_size=vocab_size,
                embedding_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout_prob=dropout_prob,
                freeze_emb=freeze_emb,
            )

        if embedding_type == "static":
            return BiGRUClassifier(
                vocab_size=vocab_size,
                embedding_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                emb_table=emb_table_arg,
                dropout_prob=dropout_prob,
                freeze_emb=freeze_emb,
            )

        if embedding_type == "contextual":
            return BiGRUClassifier(
                vocab_size=1,
                embedding_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout_prob=dropout_prob,
                freeze_emb=False,
            )

        raise ValueError(f"Unknown BiGRU embedding_type in artifacts: {embedding_type}")

    if mt_lower == "bert":
        pooling = artifacts.get("pooling", "mean")
        fine_tune = bool(artifacts.get("fine_tune", False))
        max_length = int(artifacts.get("max_length", 160))
        model_name = str(artifacts.get("model_name", "bert-base-uncased"))
        dropout = float(artifacts.get("dropout", 0.2))
        num_classes = int(artifacts["label_enc"].num_classes)

        tokenizer_name = artifacts.get("tokenizer_name")

        bert_cfg = BertEmbeddingConfig(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            dropout=0.1,
            pooling=pooling,
            fine_tune=fine_tune,
        )
        backend = BertEmbeddingBackend(bert_cfg)
        return BertClassifier(embedding_backend=backend, num_classes=num_classes, dropout=dropout)

    if mt_lower == "looped_transformer":
        return LoopedTransformer(
            vocab_size=int(artifacts["vocab_size"]),
            d_model=int(artifacts["d_model"]),
            nhead=int(artifacts["nhead"]),
            dim_feedforward=int(artifacts["dim_feedforward"]),
            num_loops=int(artifacts["num_loops"]),
            num_classes=int(artifacts["label_enc"].num_classes),
            max_seq_len=int(artifacts["max_seq_len"]),
            dropout=float(artifacts["dropout"]),
        )

    raise ValueError(f"Unsupported model_type for loading: {model_type!r}")
