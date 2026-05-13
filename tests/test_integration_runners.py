"""Integration tests for experiment runner artifact-saving behaviour.

Each test verifies that a runner produces the correct files on disk and that
the pickled artifacts dict contains the expected keys.

Fast (default): TF-IDF, BiGRU
Slow (opt-in):  BERT, LoopedTransformer — run with: pytest -m slow

Run with (from project root): pytest tests/test_integration_runners.py -v
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_glob(model_dir: str, pattern: str) -> Path:
    matches = list(Path(model_dir).glob(pattern))
    assert matches, f"No file matching {pattern!r} in {model_dir}"
    return matches[0]


def _load_artifacts(model_dir: str) -> dict:
    pkl_path = _one_glob(model_dir, "*_artifacts.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_summary(model_dir: str) -> dict:
    path = _one_glob(model_dir, "*_run_summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _assert_standard_files(model_dir: str) -> None:
    """Model dir must contain weights, summary, and artifacts."""
    _one_glob(model_dir, "*_model.pt")
    _one_glob(model_dir, "*_run_summary.json")
    _one_glob(model_dir, "*_artifacts.pkl")


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TestTFIDFRunnerArtifacts:
    def test_energy_dir_exists(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        assert Path(model_dir).is_dir()

    def test_energy_standard_files_saved(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        _assert_standard_files(model_dir)

    def test_energy_artifacts_have_vectorizer(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        assert "vectorizer" in _load_artifacts(model_dir)

    def test_energy_artifacts_have_label_enc(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        assert "label_enc" in _load_artifacts(model_dir)

    def test_energy_artifacts_flag_true(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        assert _load_artifacts(model_dir)["energy_model"] is True

    def test_energy_result_has_best_metric(self, tfidf_energy_model):
        result, _ = tfidf_energy_model
        assert isinstance(result.get("best_metric_value"), float)

    def test_energy_summary_config_has_save_dir(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        summary = _load_summary(model_dir)
        assert "save_dir" in summary.get("config", {})

    def test_damage_artifacts_flag_false(self, tfidf_damage_model):
        _, model_dir = tfidf_damage_model
        assert _load_artifacts(model_dir)["energy_model"] is False

    def test_damage_standard_files_saved(self, tfidf_damage_model):
        _, model_dir = tfidf_damage_model
        _assert_standard_files(model_dir)


# ---------------------------------------------------------------------------
# BiGRU
# ---------------------------------------------------------------------------

class TestBiGRURunnerArtifacts:
    def test_dir_exists(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert Path(model_dir).is_dir()

    def test_standard_files_saved(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        _assert_standard_files(model_dir)

    def test_artifacts_have_seq_enc(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert "seq_enc" in _load_artifacts(model_dir)

    def test_artifacts_have_vocab_enc(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert "vocab_enc" in _load_artifacts(model_dir)

    def test_artifacts_have_max_len(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert "max_len" in _load_artifacts(model_dir)

    def test_artifacts_have_energy_enc(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert "energy_enc" in _load_artifacts(model_dir)

    def test_artifacts_have_damage_enc(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert "damage_enc" in _load_artifacts(model_dir)

    def test_artifacts_flag_true(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        assert _load_artifacts(model_dir)["energy_model"] is True

    def test_best_metric_is_float(self, bigru_energy_model):
        result, _ = bigru_energy_model
        assert isinstance(result.get("best_metric_value"), float)


# ---------------------------------------------------------------------------
# BERT  (slow — needs HuggingFace weights download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBERTRunnerArtifacts:
    def test_standard_files_saved(self, bert_energy_model):
        _, model_dir = bert_energy_model
        _assert_standard_files(model_dir)

    def test_artifacts_have_label_enc(self, bert_energy_model):
        _, model_dir = bert_energy_model
        assert "label_enc" in _load_artifacts(model_dir)

    def test_artifacts_have_max_length(self, bert_energy_model):
        _, model_dir = bert_energy_model
        assert "max_length" in _load_artifacts(model_dir)

    def test_artifacts_flag_true(self, bert_energy_model):
        _, model_dir = bert_energy_model
        assert _load_artifacts(model_dir)["energy_model"] is True


# ---------------------------------------------------------------------------
# LoopedTransformer  (slow — needs HuggingFace tokeniser)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLoopedTransformerRunnerArtifacts:
    def test_standard_files_saved(self, looped_energy_model):
        _, model_dir = looped_energy_model
        _assert_standard_files(model_dir)

    def test_artifacts_have_label_enc(self, looped_energy_model):
        _, model_dir = looped_energy_model
        assert "label_enc" in _load_artifacts(model_dir)

    def test_artifacts_have_vocab_size(self, looped_energy_model):
        _, model_dir = looped_energy_model
        assert "vocab_size" in _load_artifacts(model_dir)

    def test_artifacts_have_d_model(self, looped_energy_model):
        _, model_dir = looped_energy_model
        assert "d_model" in _load_artifacts(model_dir)

    def test_artifacts_flag_true(self, looped_energy_model):
        _, model_dir = looped_energy_model
        assert _load_artifacts(model_dir)["energy_model"] is True
