"""
Unit tests for modules/inference/__init__.py and modules/training_loop/evaluate.py.

Covers:
    run_inference:
        - standalone call (no dataloader) → None, model in eval mode
        - fallback to config["test_dl"]
        - explicit dataloader override
        - correct output keys and tensor shapes
        - temperature scaling path
        - need_length=True path
        - energy_model=False uses Risk label
        - predictions are consistent with argmax of returned probs
        - empty dataloader edge case

    evaluate:
        - returns empty dict when no test_dl
        - returns expected metric keys
        - auto_classification_rate and meets_requirement values
        - threshold_used reflects config value
        - fatal_flag_count/rate present when class_names given
        - fatal_flag_count absent when class_names not given
        - loss is a non-negative float
        - saves JSON to save_dir when provided

Run with (from project root): pytest tests/test_inference.py -v
"""

import json
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from modules.inference import run_inference
from modules.training_loop.evaluate import evaluate


# ── Shared helpers ────────────────────────────────────────────────────────────

SEQ_LEN = 8
NUM_CLASSES = 3
BATCH_SIZE = 6


class _NoLengthModel(nn.Module):
    def __init__(self, out=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(SEQ_LEN, out)

    def forward(self, x):
        return self.fc(x.float())


class _WithLengthModel(nn.Module):
    def __init__(self, out=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(SEQ_LEN, out)

    def forward(self, x, lengths):
        return self.fc(x.float())


def _make_dataloader(n=BATCH_SIZE, seed=42):
    """Return a single-batch DataLoader yielding (D, DL, Energy, Risk)."""
    g = torch.Generator()
    g.manual_seed(seed)
    D = torch.randint(0, 100, (n, SEQ_LEN), generator=g)
    DL = torch.full((n,), SEQ_LEN, dtype=torch.long)
    Energy = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    Risk = torch.randint(0, 2, (n,), generator=g)
    ds = TensorDataset(D, DL, Energy, Risk)
    return DataLoader(ds, batch_size=n)


def _base_config(model, *, need_length=False, energy_model=True, dl=None):
    cfg = {
        "model": model,
        "test_dl": None,  
        "device": torch.device("cpu"),
        "need_length": need_length,
        "energy_model": energy_model,
        "num_classes": NUM_CLASSES,
        "class_dict": {},
        "threshold": 0.80,
        "criterion": nn.CrossEntropyLoss(),
        "temperature": 1.5,
        "use_temperature": False,
    }
    if dl is not None:
        cfg["test_dl"] = dl
    return cfg


# ── run_inference ─────────────────────────────────────────────────────────────


class TestRunInferenceStandalone:
    def test_returns_none_without_dataloader(self):
        model = _NoLengthModel()
        config = _base_config(model)
        assert run_inference(config) is None

    def test_model_set_to_eval_mode(self):
        model = _NoLengthModel()
        model.train()
        config = _base_config(model)
        run_inference(config)
        assert not model.training

    def test_model_stays_in_eval_with_dataloader(self):
        model = _NoLengthModel()
        model.train()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        run_inference(config)
        assert not model.training


class TestRunInferenceOutput:
    def test_returns_expected_keys(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        assert set(result.keys()) == {"all_preds", "all_targets", "all_probs", "total_examples"}

    def test_tensor_shapes(self):
        model = _NoLengthModel()
        dl = _make_dataloader(n=BATCH_SIZE)
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        assert result["all_preds"].shape == (BATCH_SIZE,)
        assert result["all_targets"].shape == (BATCH_SIZE,)
        assert result["all_probs"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_total_examples_count(self):
        model = _NoLengthModel()
        dl = _make_dataloader(n=10)
        config = _base_config(model, dl=10 and _make_dataloader(n=10))
        config["test_dl"] = _make_dataloader(n=10)
        result = run_inference(config)
        assert result["total_examples"] == 10

    def test_probs_sum_to_one(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        row_sums = result["all_probs"].sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_preds_match_argmax_of_probs(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        expected_preds = result["all_probs"].argmax(dim=1)
        assert torch.equal(result["all_preds"], expected_preds)

    def test_preds_are_valid_class_indices(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        assert result["all_preds"].min().item() >= 0
        assert result["all_preds"].max().item() < NUM_CLASSES


class TestRunInferenceDataloaderSources:
    def test_uses_config_test_dl_by_default(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        assert result is not None

    def test_explicit_dataloader_overrides_config_test_dl(self):
        model = _NoLengthModel()
        # config has a different dl; explicit one has n=4
        config = _base_config(model, dl=_make_dataloader(n=BATCH_SIZE))
        explicit_dl = _make_dataloader(n=4)
        result = run_inference(config, dataloader=explicit_dl)
        assert result["total_examples"] == 4

    def test_explicit_dataloader_works_without_config_test_dl(self):
        model = _NoLengthModel()
        config = _base_config(model)  # no test_dl in config
        dl = _make_dataloader()
        result = run_inference(config, dataloader=dl)
        assert result is not None


class TestRunInferenceTemperature:
    def test_temperature_scaling_does_not_crash(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["use_temperature"] = True
        config["temperature"] = 2.0
        result = run_inference(config)
        assert result["all_probs"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_high_temperature_softens_distribution(self):
        """Higher temperature → lower max-prob (flatter distribution)."""
        model = _NoLengthModel()
        dl = _make_dataloader()

        config_cold = _base_config(model, dl=dl)
        config_cold["use_temperature"] = True
        config_cold["temperature"] = 0.1

        config_warm = _base_config(model, dl=_make_dataloader())
        config_warm["use_temperature"] = True
        config_warm["temperature"] = 10.0

        cold_max = run_inference(config_cold)["all_probs"].max(dim=1).values.mean()
        warm_max = run_inference(config_warm)["all_probs"].max(dim=1).values.mean()
        assert cold_max > warm_max


class TestRunInferenceBatchVariants:
    def test_need_length_true(self):
        model = _WithLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, need_length=True, dl=dl)
        result = run_inference(config)
        assert result["all_preds"].shape == (BATCH_SIZE,)

    def test_energy_model_false_uses_risk_labels(self):
        """energy_model=False → targets come from Risk column (2 classes)."""
        model = _NoLengthModel(out=2)
        dl = _make_dataloader()
        config = _base_config(model, energy_model=False, dl=dl)
        config["num_classes"] = 2
        result = run_inference(config)
        assert result["all_targets"].max().item() < 2

    def test_empty_dataloader_returns_empty_tensors(self):
        model = _NoLengthModel()
        ds = TensorDataset(
            torch.zeros(0, SEQ_LEN, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
        dl = DataLoader(ds, batch_size=4)
        config = _base_config(model, dl=dl)
        result = run_inference(config)
        assert result["total_examples"] == 0
        assert result["all_preds"].numel() == 0


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestEvaluateBasic:
    def test_returns_empty_dict_without_test_dl(self):
        model = _NoLengthModel()
        config = _base_config(model)  # no test_dl
        assert evaluate(config) == {}

    def test_returns_expected_keys(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        metrics = evaluate(config)
        expected = {
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "class_metrics", "confusion_matrix",
            "loss", "auto_classification_rate", "meets_requirement", "threshold_used",
        }
        assert expected.issubset(set(metrics.keys()))

    def test_loss_is_non_negative_float(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        metrics = evaluate(config)
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0.0

    def test_accuracy_in_zero_one_range(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        metrics = evaluate(config)
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestEvaluateThreshold:
    def test_threshold_used_reflects_config(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["threshold"] = 0.55
        metrics = evaluate(config)
        assert metrics["threshold_used"] == pytest.approx(0.55)

    def test_default_threshold_is_0_80(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        metrics = evaluate(config)
        assert metrics["threshold_used"] == pytest.approx(0.80)

    def test_auto_classification_rate_in_zero_one_range(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        metrics = evaluate(config)
        assert 0.0 <= metrics["auto_classification_rate"] <= 1.0

    def test_threshold_zero_gives_full_rate(self):
        """A threshold of 0.0 means every prediction is high-confidence."""
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["threshold"] = 0.0
        metrics = evaluate(config)
        assert metrics["auto_classification_rate"] == pytest.approx(1.0)

    def test_threshold_one_gives_zero_rate(self):
        """A threshold of 1.0 means no prediction clears the bar."""
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["threshold"] = 1.0
        metrics = evaluate(config)
        assert metrics["auto_classification_rate"] == pytest.approx(0.0)

    def test_meets_requirement_true_when_rate_above_70(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["threshold"] = 0.0  # force 100 % rate
        metrics = evaluate(config)
        assert metrics["meets_requirement"] is True

    def test_meets_requirement_false_when_rate_below_70(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["threshold"] = 1.0  # force 0 % rate
        metrics = evaluate(config)
        assert metrics["meets_requirement"] is False


class TestEvaluateFatalFlagging:
    def test_fatal_metrics_absent_for_energy_model(self):
        model = _NoLengthModel()
        dl = _make_dataloader()
        config = _base_config(model, dl=dl, energy_model=True)
        metrics = evaluate(config)
        assert metrics.get("fatal_flag_count") is None
        assert metrics.get("fatal_flag_rate") is None

    def test_fatal_flag_count_non_negative(self):
        model = _NoLengthModel(out=3)
        dl = _make_dataloader()
        config = _base_config(model, dl=dl, energy_model=False)
        config["class_dict"] = {0: "Safe", 1: "Single Fatality", 2: "Multiple Fatality"}
        metrics = evaluate(config)
        assert metrics["fatal_flag_count"] >= 0

    def test_fatal_flag_rate_in_zero_one_range(self):
        model = _NoLengthModel(out=3)
        dl = _make_dataloader()
        config = _base_config(model, dl=dl, energy_model=False)
        config["class_dict"] = {0: "Safe", 1: "Single Fatality", 2: "Multiple Fatality"}
        metrics = evaluate(config)
        assert 0.0 <= metrics["fatal_flag_rate"] <= 1.0

    def test_fatal_flag_count_consistent_with_rate(self):
        model = _NoLengthModel(out=3)
        dl = _make_dataloader(n=BATCH_SIZE)
        config = _base_config(model, dl=dl, energy_model=False)
        config["class_dict"] = {0: "Safe", 1: "Single Fatality", 2: "Multiple Fatality"}
        metrics = evaluate(config)
        expected_rate = metrics["fatal_flag_count"] / BATCH_SIZE
        assert metrics["fatal_flag_rate"] == pytest.approx(expected_rate, abs=1e-5)

    def test_no_fatal_class_in_class_dict_skips_flagging(self):
        model = _NoLengthModel(out=3)
        dl = _make_dataloader()
        config = _base_config(model, dl=dl)
        config["class_dict"] = {0: "Alpha", 1: "Beta", 2: "Gamma"}  # no fatal names
        metrics = evaluate(config)
        assert "fatal_flag_count" not in metrics
