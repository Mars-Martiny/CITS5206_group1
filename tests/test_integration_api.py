"""Integration tests for the api/ module.

Covers api.train, api.load_model, api.infer, api.get_leaderboard,
and api.get_model_details using real CSV splits and actual model runs.

Fast (default): TF-IDF, BiGRU
Slow (opt-in):  BERT, LoopedTransformer — run with: pytest -m slow

Run with (from project root): pytest tests/test_integration_api.py -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api import get_leaderboard, get_model_details, infer, load_model, train


# ---------------------------------------------------------------------------
# api.train
# ---------------------------------------------------------------------------

class TestAPITrain:
    def test_tfidf_returns_dict(self, tfidf_energy_model):
        result, _ = tfidf_energy_model
        assert isinstance(result, dict)

    def test_tfidf_config_has_save_dir(self, tfidf_energy_model):
        result, _ = tfidf_energy_model
        assert "save_dir" in result["config"]

    def test_bigru_returns_dict(self, bigru_energy_model):
        result, _ = bigru_energy_model
        assert isinstance(result, dict)

    def test_bigru_config_has_save_dir(self, bigru_energy_model):
        result, _ = bigru_energy_model
        assert "save_dir" in result["config"]

    def test_invalid_model_type_raises(self, raw_csv):
        with pytest.raises(ValueError, match="model_type"):
            train(
                train_path=str(raw_csv["train"]),
                valid_path=str(raw_csv["valid"]),
                test_path=str(raw_csv["test"]),
                model_type="unknown",
                architecture="tf_idf",
            )

    def test_invalid_architecture_raises(self, raw_csv):
        with pytest.raises(ValueError, match="architecture"):
            train(
                train_path=str(raw_csv["train"]),
                valid_path=str(raw_csv["valid"]),
                test_path=str(raw_csv["test"]),
                model_type="energy",
                architecture="nonexistent_arch",
            )


# ---------------------------------------------------------------------------
# api.load_model
# ---------------------------------------------------------------------------

class TestAPILoadModel:
    def test_tfidf_bundle_keys(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        bundle = load_model(model_dir)
        for key in ("model", "model_type", "energy_model", "artifacts", "label_enc",
                    "class_dict", "config", "device", "num_classes"):
            assert key in bundle, f"bundle missing {key!r}"

    def test_tfidf_energy_flag_true(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        bundle = load_model(model_dir)
        assert bundle["energy_model"] is True

    def test_tfidf_damage_flag_false(self, tfidf_damage_model):
        _, model_dir = tfidf_damage_model
        bundle = load_model(model_dir)
        assert bundle["energy_model"] is False

    def test_tfidf_model_is_in_eval(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        bundle = load_model(model_dir)
        assert not bundle["model"].training

    def test_tfidf_num_classes_positive(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        bundle = load_model(model_dir)
        assert bundle["num_classes"] > 0

    def test_tfidf_class_dict_is_dict(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        bundle = load_model(model_dir)
        assert isinstance(bundle["class_dict"], dict)

    def test_bigru_bundle_keys(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        bundle = load_model(model_dir)
        for key in ("model", "model_type", "energy_model", "artifacts", "label_enc"):
            assert key in bundle, f"bundle missing {key!r}"

    def test_bigru_model_is_in_eval(self, bigru_energy_model):
        _, model_dir = bigru_energy_model
        bundle = load_model(model_dir)
        assert not bundle["model"].training

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/this_dir_does_not_exist_xyz")

    @pytest.mark.slow
    def test_bert_bundle_keys(self, bert_energy_model):
        _, model_dir = bert_energy_model
        bundle = load_model(model_dir)
        for key in ("model", "model_type", "energy_model", "label_enc"):
            assert key in bundle

    @pytest.mark.slow
    def test_looped_bundle_keys(self, looped_energy_model):
        _, model_dir = looped_energy_model
        bundle = load_model(model_dir)
        for key in ("model", "model_type", "energy_model", "label_enc"):
            assert key in bundle


# ---------------------------------------------------------------------------
# api.infer
# ---------------------------------------------------------------------------

class TestAPIInfer:
    def test_energy_only_returns_dataframe(self, tfidf_energy_model, raw_csv):
        _, model_dir = tfidf_energy_model
        result = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        assert isinstance(result, pd.DataFrame)

    def test_energy_only_expected_columns(self, tfidf_energy_model, raw_csv):
        _, model_dir = tfidf_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        for col in ("predicted_energy_type", "energy_confidence", "energy_score",
                    "energy_action_required"):
            assert col in df.columns, f"missing column {col!r}"

    def test_energy_only_no_damage_columns(self, tfidf_energy_model, raw_csv):
        _, model_dir = tfidf_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        for col in ("predicted_damage_potential", "fatal_flag"):
            assert col not in df.columns

    def test_energy_only_row_count_preserved(self, tfidf_energy_model, raw_csv):
        _, model_dir = tfidf_energy_model
        n_test = len(pd.read_csv(raw_csv["test"]))
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        assert len(df) == n_test

    def test_damage_only_expected_columns(self, tfidf_damage_model, raw_csv):
        _, model_dir = tfidf_damage_model
        df = infer(dataset_path=str(raw_csv["test"]), damage_model_dir=model_dir)
        for col in ("predicted_damage_potential", "damage_confidence", "damage_score",
                    "fatal_flag", "damage_action_required"):
            assert col in df.columns

    def test_damage_only_no_energy_columns(self, tfidf_damage_model, raw_csv):
        _, model_dir = tfidf_damage_model
        df = infer(dataset_path=str(raw_csv["test"]), damage_model_dir=model_dir)
        assert "predicted_energy_type" not in df.columns

    def test_both_models_all_columns(self, tfidf_energy_model, tfidf_damage_model, raw_csv):
        _, energy_dir = tfidf_energy_model
        _, damage_dir = tfidf_damage_model
        df = infer(
            dataset_path=str(raw_csv["test"]),
            energy_model_dir=energy_dir,
            damage_model_dir=damage_dir,
        )
        expected = [
            "predicted_energy_type", "energy_confidence", "energy_score",
            "predicted_damage_potential", "damage_confidence", "damage_score",
            "fatal_flag", "energy_action_required", "damage_action_required",
        ]
        for col in expected:
            assert col in df.columns, f"missing column {col!r}"

    def test_score_values_are_valid_tiers(self, tfidf_energy_model, raw_csv):
        _, model_dir = tfidf_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        valid_tiers = {"HIGH", "MEDIUM", "LOW"}
        unique = set(df["energy_score"].dropna().astype(str).tolist())
        assert unique.issubset(valid_tiers), f"unexpected tier values: {unique - valid_tiers}"

    def test_confidence_is_float_between_0_and_1(self, tfidf_energy_model, raw_csv):
        _, model_dir = tfidf_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        vals = df["energy_confidence"].dropna().astype(float)
        assert (vals >= 0.0).all() and (vals <= 1.0).all()

    def test_fatal_flag_values(self, tfidf_damage_model, raw_csv):
        _, model_dir = tfidf_damage_model
        df = infer(dataset_path=str(raw_csv["test"]), damage_model_dir=model_dir)
        unique = set(df["fatal_flag"].dropna().astype(str).tolist())
        assert unique.issubset({"YES", "NO"})

    def test_output_csv_written(self, tfidf_energy_model, raw_csv, tmp_path):
        _, model_dir = tfidf_energy_model
        out = tmp_path / "preds.csv"
        infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir,
              output_path=str(out))
        assert out.exists()
        assert len(pd.read_csv(out)) > 0

    def test_no_model_raises_value_error(self, raw_csv):
        with pytest.raises(ValueError, match="At least one"):
            infer(dataset_path=str(raw_csv["test"]))

    @pytest.mark.slow
    def test_bert_infer_energy_only(self, bert_energy_model, raw_csv):
        _, model_dir = bert_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        assert "predicted_energy_type" in df.columns

    @pytest.mark.slow
    def test_looped_infer_energy_only(self, looped_energy_model, raw_csv):
        _, model_dir = looped_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        assert "predicted_energy_type" in df.columns

    @pytest.mark.slow
    def test_bigru_infer_energy_only(self, bigru_energy_model, raw_csv):
        _, model_dir = bigru_energy_model
        df = infer(dataset_path=str(raw_csv["test"]), energy_model_dir=model_dir)
        assert "predicted_energy_type" in df.columns


# ---------------------------------------------------------------------------
# api.get_leaderboard / api.get_model_details
# ---------------------------------------------------------------------------

class TestAPIMetrics:
    def test_get_leaderboard_returns_dataframe(self):
        df = get_leaderboard()
        assert isinstance(df, pd.DataFrame)

    def test_leaderboard_has_rows(self):
        df = get_leaderboard()
        assert len(df) > 0

    def test_leaderboard_sort_ascending(self):
        df = get_leaderboard(sort_by="val_f1_macro", ascending=True)
        vals = df["val_f1_macro"].dropna().tolist()
        assert vals == sorted(vals)

    def test_leaderboard_sort_descending(self):
        df = get_leaderboard(sort_by="val_f1_macro", ascending=False)
        vals = df["val_f1_macro"].dropna().tolist()
        assert vals == sorted(vals, reverse=True)

    def test_leaderboard_energy_filter(self):
        df = get_leaderboard(model_type_filter="energy")
        assert len(df) > 0

    def test_leaderboard_damage_filter(self):
        df = get_leaderboard(model_type_filter="damage")
        assert len(df) > 0

    def test_leaderboard_architecture_filter_tfidf(self):
        df = get_leaderboard(architecture_filter="tf_idf")
        assert len(df) > 0

    def test_leaderboard_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            get_leaderboard(model_type_filter="invalid")

    def test_get_model_details_returns_dict(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        details = get_model_details(model_dir)
        assert isinstance(details, dict)

    def test_get_model_details_has_config(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        details = get_model_details(model_dir)
        assert "config" in details

    def test_get_model_details_has_best_metric(self, tfidf_energy_model):
        _, model_dir = tfidf_energy_model
        details = get_model_details(model_dir)
        assert "best_metric_value" in details

    def test_get_model_details_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            get_model_details("/tmp/nonexistent_model_dir_xyz")
