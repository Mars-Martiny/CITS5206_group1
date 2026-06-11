"""Tests for retrain / fine-tuning functionality.

Drop this file into the repository as:

    tests/test_retrain.py

These tests assume the retrain implementation exposes:

    api.retrain.retrain(...)
    api.retrain.register_retrainer(...)
    api.retrain._RETRAINERS

The TF-IDF tests use the existing raw_csv fixture from tests/conftest.py on the
dev branch. The BERT test is dispatcher-level and mocked, so it does not
download Hugging Face weights or run a real BERT training loop.

The tests write model outputs to a pytest temporary directory where possible.
They also remove any project-level trained_models/*pytest_* directories created
by implementations that ignore the supplied parent_dir.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _remove_project_pytest_model_dirs() -> None:
    """Remove stale pytest-created model directories from the project tree only.

    This intentionally targets names containing "pytest_" so it does not delete
    real trained models that a developer may have saved under trained_models/.
    """
    root = Path("trained_models")
    if not root.exists():
        return

    for child in root.iterdir():
        if child.is_dir() and "pytest_" in child.name:
            shutil.rmtree(child, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def cleanup_project_pytest_models():
    """Clean project-level pytest model artifacts before and after this module."""
    _remove_project_pytest_model_dirs()
    yield
    _remove_project_pytest_model_dirs()


@pytest.fixture(scope="module")
def model_output_dir(tmp_path_factory):
    """Temporary parent directory for model artifacts produced by these tests."""
    root = tmp_path_factory.mktemp("retrain_models")
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def base_tfidf_energy_model(raw_csv, model_output_dir):
    """Train a small TF-IDF base model for retrain tests.

    This avoids using the session-scoped tfidf_energy_model fixture because that
    fixture writes to the default trained_models/ directory and may update the
    leaderboard.
    """
    from api.train import train

    result = train(
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        model_type="energy",
        architecture="tf_idf",
        train_config={
            "epochs": 1,
            "patience": 1,
            "run_name": "pytest_base_tfidf_energy",
            "parent_dir": str(model_output_dir),
            "log_leaderboard": False,
            "verbose": False,
        },
    )
    return result, Path(result["config"]["save_dir"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_glob(model_dir: str | Path, pattern: str) -> Path:
    root = Path(model_dir)
    matches = sorted(root.glob(pattern))
    assert matches, f"No file matching {pattern!r} under {root}"
    return matches[0]


def _assert_standard_files(model_dir: str | Path) -> None:
    """A retrained run must preserve the existing exported-model contract."""
    _one_glob(model_dir, "*_model.pt")
    _one_glob(model_dir, "*_run_summary.json")
    _one_glob(model_dir, "*_artifacts.pkl")


def _load_artifacts(model_dir: str | Path) -> dict[str, Any]:
    artifacts_path = _one_glob(model_dir, "*_artifacts.pkl")
    with open(artifacts_path, "rb") as f:
        return pickle.load(f)


def _result_save_dir(result: dict[str, Any]) -> Path:
    return Path(result["config"]["save_dir"])


def _fake_embedding_matrix(
    vocab: dict[str, int],
    *,
    model_name: str = "pytest/fake",
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Small deterministic embedding matrix used to avoid Hugging Face downloads."""
    size = max(vocab.values()) + 1 if vocab else 1
    embed_dim = 4

    matrix = torch.arange(size * embed_dim, dtype=torch.float32).reshape(size, embed_dim)
    if device is not None:
        matrix = matrix.to(device)
    return matrix


def _get_saved_embedding_matrix(artifacts: dict[str, Any]) -> torch.Tensor | None:
    """Accept a few reasonable key names for saved TF-IDF embedding artifacts."""
    for key in ("embedding_matrix", "tfidf_embedding_matrix", "embeddings"):
        if key in artifacts:
            value = artifacts[key]
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value)
    return None


# ---------------------------------------------------------------------------
# TF-IDF refresh retraining
# ---------------------------------------------------------------------------

def test_tfidf_refresh_retrains_and_updates_embedding_artifacts(
    raw_csv,
    base_tfidf_energy_model,
    model_output_dir,
    monkeypatch,
):
    """Refresh mode should rebuild TF-IDF artifacts and save dense embeddings.

    This checks the "saved embeddings" requirement for
    feature_representation="tfidf_embed_avg". The SafetyBERT embedding builder
    is monkeypatched so the test stays fast and deterministic.

    input_dim is intentionally treated as optional here. If the implementation
    saves it, we validate it. If it does not, this test will not fail with a raw
    KeyError.
    """
    from api.loader import load_model
    from api.retrain import retrain
    import modules.embedding.safety_bert_static as safety_static

    _, old_model_dir = base_tfidf_energy_model

    monkeypatch.setattr(
        safety_static,
        "get_safety_bert_embedding_matrix",
        _fake_embedding_matrix,
    )

    result = retrain(
        model_dir=str(old_model_dir),
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="refresh",
        train_config={
            "epochs": 1,
            "patience": 1,
            "run_name": "pytest_tfidf_refresh_embed",
            "parent_dir": str(model_output_dir),
            "feature_representation": "tfidf_embed_avg",
            "embedding_model_name": "pytest/fake",
            "log_leaderboard": False,
            "verbose": False,
        },
    )

    new_model_dir = _result_save_dir(result)

    assert new_model_dir.is_dir()
    assert new_model_dir != Path(old_model_dir)
    _assert_standard_files(new_model_dir)

    artifacts = _load_artifacts(new_model_dir)

    assert artifacts["energy_model"] is True
    assert artifacts["feature_representation"] == "tfidf_embed_avg"
    assert artifacts["embedding_model_name"] == "pytest/fake"
    assert "vectorizer" in artifacts
    assert "label_enc" in artifacts

    embedding_matrix = _get_saved_embedding_matrix(artifacts)
    assert embedding_matrix is not None, (
        "Expected TF-IDF embedding-average retraining to save the embedding "
        "matrix in *_artifacts.pkl under one of: embedding_matrix, "
        "tfidf_embedding_matrix, embeddings."
    )
    assert embedding_matrix.ndim == 2
    assert embedding_matrix.shape[1] == 4

    saved_input_dim = artifacts.get("input_dim")
    if saved_input_dim is not None:
        assert int(saved_input_dim) == int(embedding_matrix.shape[1])

        # Only run the loader round-trip when input_dim is available. Without it,
        # implementations that reconstruct TF-IDF from len(vectorizer.vocab) can
        # fail even though the embedding artifact itself was correctly saved.
        bundle = load_model(str(new_model_dir))
        assert bundle["model_type"].lower() == "tf_idf"
        assert bundle["num_classes"] == artifacts["label_enc"].num_classes


# ---------------------------------------------------------------------------
# TF-IDF continue retraining
# ---------------------------------------------------------------------------

def test_tfidf_continue_keeps_existing_vectorizer_and_saves_new_artifacts(
    raw_csv,
    base_tfidf_energy_model,
    model_output_dir,
):
    """Continue mode should train from the previous TF-IDF checkpoint.

    For TF-IDF, continuing from a checkpoint should keep the previous vectorizer
    fixed because changing the vocabulary changes the classifier input shape.
    """
    from api.retrain import retrain

    _, old_model_dir = base_tfidf_energy_model
    old_artifacts = _load_artifacts(old_model_dir)
    old_vocab = dict(old_artifacts["vectorizer"].vocab)

    result = retrain(
        model_dir=str(old_model_dir),
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="continue",
        train_config={
            "epochs": 1,
            "patience": 1,
            "run_name": "pytest_tfidf_continue",
            "parent_dir": str(model_output_dir),
            "log_leaderboard": False,
            "verbose": False,
        },
    )

    new_model_dir = _result_save_dir(result)
    new_artifacts = _load_artifacts(new_model_dir)

    assert new_model_dir.is_dir()
    assert new_model_dir != Path(old_model_dir)
    _assert_standard_files(new_model_dir)

    assert dict(new_artifacts["vectorizer"].vocab) == old_vocab
    assert new_artifacts["label_enc"].id_to_label == old_artifacts["label_enc"].id_to_label
    assert new_artifacts.get("retrain_mode") == "continue"


def test_tfidf_continue_rejects_unseen_labels(
    raw_csv,
    base_tfidf_energy_model,
    model_output_dir,
    tmp_path,
):
    """Continue mode should not silently add new output classes.

    Adding new classes requires rebuilding the label encoder and classifier head,
    so the safer expected behaviour is to raise on unseen labels.
    """
    from api.retrain import retrain

    _, old_model_dir = base_tfidf_energy_model

    bad_train = pd.read_csv(raw_csv["train"])
    bad_train.loc[0, "Energy Type"] = "Chemical"

    bad_train_path = tmp_path / "bad_train.csv"
    bad_train.to_csv(bad_train_path, index=False)

    with pytest.raises(ValueError, match="Unknown label"):
        retrain(
            model_dir=str(old_model_dir),
            train_path=str(bad_train_path),
            valid_path=str(raw_csv["valid"]),
            test_path=str(raw_csv["test"]),
            mode="continue",
            train_config={
                "epochs": 1,
                "patience": 1,
                "run_name": "pytest_tfidf_continue_bad_label",
                "parent_dir": str(model_output_dir),
                "log_leaderboard": False,
                "verbose": False,
            },
        )


# ---------------------------------------------------------------------------
# BERT dispatch without slow model downloads
# ---------------------------------------------------------------------------

def test_retrain_auto_dispatches_bert_to_continue_without_huggingface(
    raw_csv,
    monkeypatch,
):
    """mode='auto' should route BERT models to continue-mode retraining.

    This is intentionally mocked. It tests the API dispatch contract without
    downloading BERT weights or running a slow training loop.
    """
    import api.retrain as retrain_module

    captured: dict[str, Any] = {}

    class DummyLabelEncoder:
        num_classes = 2
        id_to_label = {0: "Electrical", 1: "Vehicular"}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    def fake_load_model(model_dir: str) -> dict[str, Any]:
        return {
            "model": DummyModel(),
            "model_type": "bert",
            "energy_model": True,
            "artifacts": {
                "energy_model": True,
                "text_col": "description",
                "model_name": "bert-base-uncased",
                "tokenizer_name": "bert-base-uncased",
                "max_length": 16,
                "pooling": "mean",
                "fine_tune": True,
                "batch_size": 2,
            },
            "label_enc": DummyLabelEncoder(),
            "config": {
                "save_dir": model_dir,
                "run_name": "old_bert_run",
                "model_type": "BERT",
            },
            "device": torch.device("cpu"),
            "num_classes": 2,
            "class_dict": {0: "Electrical", 1: "Vehicular"},
        }

    def fake_bert_retrainer(
        bundle: dict[str, Any],
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: str,
        train_config: dict[str, Any],
    ) -> dict[str, Any]:
        captured["bundle"] = bundle
        captured["train_rows"] = len(train_df)
        captured["valid_rows"] = len(valid_df)
        captured["test_rows"] = len(test_df)
        captured["text_col"] = text_col
        captured["train_config"] = train_config
        return {
            "config": {"save_dir": "trained_models/pytest_mock_bert_continue"},
            "best_metric_value": 0.5,
        }

    monkeypatch.setattr(retrain_module, "load_model", fake_load_model)
    monkeypatch.setitem(retrain_module._RETRAINERS, "bert", fake_bert_retrainer)

    result = retrain_module.retrain(
        model_dir="trained_models/old_bert_run",
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="auto",
        train_config={"epochs": 1, "learning_rate": 1e-5},
    )

    assert result["best_metric_value"] == 0.5
    assert captured["bundle"]["model_type"] == "bert"
    assert captured["text_col"] == "description"
    assert captured["train_config"]["epochs"] == 1
    assert captured["train_config"]["learning_rate"] == 1e-5
    assert captured["train_rows"] > 0
    assert captured["valid_rows"] > 0
    assert captured["test_rows"] > 0


# ---------------------------------------------------------------------------
# Extensibility / future architectures
# ---------------------------------------------------------------------------

def test_retrain_refresh_delegates_to_existing_train_api_for_future_models(
    raw_csv,
    monkeypatch,
):
    """Refresh mode should be architecture-agnostic.

    Future models should be able to retrain by using the existing api.train.train
    dispatcher instead of requiring custom continuation logic immediately.
    """
    import api.retrain as retrain_module

    captured: dict[str, Any] = {}

    def fake_load_model(model_dir: str) -> dict[str, Any]:
        return {
            "model_type": "looped_transformer",
            "energy_model": False,
            "artifacts": {"text_col": "description"},
            "config": {
                "save_dir": model_dir,
                "run_name": "old_looped_damage",
                "save_name": "old_looped_damage",
            },
        }

    def fake_train_from_splits(
        train_path: str,
        valid_path: str,
        test_path: str,
        model_type: str,
        architecture: str,
        train_config: dict | None = None,
        text_col: str = "description",
    ) -> dict[str, Any]:
        captured.update(
            {
                "train_path": train_path,
                "valid_path": valid_path,
                "test_path": test_path,
                "model_type": model_type,
                "architecture": architecture,
                "train_config": train_config or {},
                "text_col": text_col,
            }
        )
        return {
            "config": {"save_dir": "trained_models/pytest_mock_looped_refresh"},
            "best_metric_value": 0.25,
        }

    monkeypatch.setattr(retrain_module, "load_model", fake_load_model)
    monkeypatch.setattr(retrain_module, "train_from_splits", fake_train_from_splits)

    result = retrain_module.retrain(
        model_dir="trained_models/old_looped_damage",
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="refresh",
        train_config={"epochs": 1},
    )

    assert result["best_metric_value"] == 0.25
    assert captured["model_type"] == "damage"
    assert captured["architecture"] == "looped_transformer"
    assert captured["text_col"] == "description"
    assert captured["train_config"]["epochs"] == 1
