"""Integration tests for cli.py Click commands.

Uses Click's CliRunner to invoke commands in-process without spawning a
subprocess. Trained model directories from conftest session fixtures are
passed directly to infer and metrics commands.

Fast (default): TF-IDF training, infer, metrics
Slow (opt-in):  BERT, LoopedTransformer — run with: pytest -m slow

Run with (from project root): pytest tests/test_integration_cli.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

class TestCLITrain:
    def test_tfidf_energy_exits_zero(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "tf_idf",
            "--epochs", "2",
        ])
        assert result.exit_code == 0, result.output

    def test_tfidf_energy_prints_artifacts_dir(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "tf_idf",
            "--epochs", "2",
        ])
        assert "Artifacts directory:" in result.output

    def test_tfidf_damage_exits_zero(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "damage",
            "--architecture", "tf_idf",
            "--epochs", "2",
        ])
        assert result.exit_code == 0, result.output

    def test_bigru_exits_zero(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "bigru",
            "--epochs", "2",
        ])
        assert result.exit_code == 0, result.output

    def test_invalid_model_type_shows_error(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "oops",
            "--architecture", "tf_idf",
        ])
        assert result.exit_code != 0

    def test_invalid_architecture_shows_error(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "svm",
        ])
        assert result.exit_code != 0

    def test_prints_best_metric(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "tf_idf",
            "--epochs", "2",
        ])
        assert "Best tracked metric:" in result.output

    @pytest.mark.slow
    def test_bert_exits_zero(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "bert",
            "--epochs", "1",
        ])
        assert result.exit_code == 0, result.output

    @pytest.mark.slow
    def test_looped_transformer_exits_zero(self, runner, raw_csv):
        result = runner.invoke(cli, [
            "train",
            "--train", str(raw_csv["train"]),
            "--valid", str(raw_csv["valid"]),
            "--test",  str(raw_csv["test"]),
            "--model-type", "energy",
            "--architecture", "looped_transformer",
            "--epochs", "1",
        ])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# infer command
# ---------------------------------------------------------------------------

class TestCLIInfer:
    def test_energy_only_exits_zero(self, runner, raw_csv, tfidf_energy_model, tmp_path):
        _, model_dir = tfidf_energy_model
        out = tmp_path / "preds.csv"
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(out),
        ])
        assert result.exit_code == 0, result.output

    def test_energy_only_prints_rows_scored(self, runner, raw_csv, tfidf_energy_model, tmp_path):
        _, model_dir = tfidf_energy_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(tmp_path / "preds.csv"),
        ])
        assert "Rows scored:" in result.output

    def test_energy_only_prints_tier_counts(self, runner, raw_csv, tfidf_energy_model, tmp_path):
        _, model_dir = tfidf_energy_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(tmp_path / "preds.csv"),
        ])
        assert "Energy tier counts:" in result.output

    def test_both_models_prints_fatal_flagged(
        self, runner, raw_csv, tfidf_energy_model, tfidf_damage_model, tmp_path
    ):
        _, energy_dir = tfidf_energy_model
        _, damage_dir = tfidf_damage_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", energy_dir,
            "--damage-model", damage_dir,
            "--output", str(tmp_path / "both.csv"),
        ])
        assert result.exit_code == 0, result.output
        assert "Fatal-flagged rows:" in result.output

    def test_no_model_exits_nonzero(self, runner, raw_csv, tmp_path):
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--output", str(tmp_path / "preds.csv"),
        ])
        assert result.exit_code != 0

    def test_output_csv_created(self, runner, raw_csv, tfidf_energy_model, tmp_path):
        _, model_dir = tfidf_energy_model
        out = tmp_path / "preds.csv"
        runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(out),
        ])
        assert out.exists()

    def test_damage_only_prints_damage_tier_counts(
        self, runner, raw_csv, tfidf_damage_model, tmp_path
    ):
        _, model_dir = tfidf_damage_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--damage-model", model_dir,
            "--output", str(tmp_path / "dmg.csv"),
        ])
        assert "Damage tier counts:" in result.output

    @pytest.mark.slow
    def test_bigru_infer_exits_zero(self, runner, raw_csv, bigru_energy_model, tmp_path):
        _, model_dir = bigru_energy_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(tmp_path / "bigru.csv"),
        ])
        assert result.exit_code == 0, result.output

    @pytest.mark.slow
    def test_bert_infer_exits_zero(self, runner, raw_csv, bert_energy_model, tmp_path):
        _, model_dir = bert_energy_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(tmp_path / "bert.csv"),
        ])
        assert result.exit_code == 0, result.output

    @pytest.mark.slow
    def test_looped_infer_exits_zero(self, runner, raw_csv, looped_energy_model, tmp_path):
        _, model_dir = looped_energy_model
        result = runner.invoke(cli, [
            "infer",
            "--dataset", str(raw_csv["test"]),
            "--energy-model", model_dir,
            "--output", str(tmp_path / "looped.csv"),
        ])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# metrics command
# ---------------------------------------------------------------------------

class TestCLIMetrics:
    def test_leaderboard_exits_zero(self, runner):
        result = runner.invoke(cli, ["metrics", "--top", "5"])
        assert result.exit_code == 0, result.output

    def test_leaderboard_ascending_exits_zero(self, runner):
        result = runner.invoke(cli, ["metrics", "--top", "5", "--ascending"])
        assert result.exit_code == 0, result.output

    def test_leaderboard_model_type_filter(self, runner):
        result = runner.invoke(cli, ["metrics", "--model-type", "energy", "--top", "5"])
        assert result.exit_code == 0, result.output

    def test_leaderboard_architecture_filter(self, runner):
        result = runner.invoke(cli, ["metrics", "--architecture", "tf_idf", "--top", "5"])
        assert result.exit_code == 0, result.output

    def test_model_dir_flag_prints_json(self, runner, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        result = runner.invoke(cli, ["metrics", "--model-dir", model_dir])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "config" in parsed

    def test_model_dir_has_best_metric_in_output(self, runner, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        result = runner.invoke(cli, ["metrics", "--model-dir", model_dir])
        assert "best_metric_value" in result.output
