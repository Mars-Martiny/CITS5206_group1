"""
Unit tests for modules/training_loop utility functions.

Covers pure/side-effect-free functions that don't require a full training run:
    - _is_better
    - _serialise_value
    - _compute_classification_metrics
    - _unpack_batch
    - _get_learning_rates

Run with (from the project root): pytest tests/test_training_loop.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from modules.training_loop.utility import (
    _is_better,
    _serialise_value,
    _unpack_batch,
    _get_learning_rates,
)

# Import directly from the submodule to avoid pulling in run_saving (matplotlib)
# via modules/training_loop/__init__.py
from modules.training_loop.metrics import _compute_classification_metrics
from modules.training_loop.loss import get_loss_function, FocalLoss
from modules.training_loop.imbalance import compute_class_weights, make_weighted_sampler

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=8, out_features=3, use_length=False):
    """Minimal linear model compatible with _unpack_batch.
    Expects D of shape (batch, seq_len) and returns logits of shape (batch, out_features).
    """
    if use_length:

        class _WithLength(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(seq_len, out_features)

            def forward(self, x, lengths):
                return self.fc(x.float())

        return _WithLength()
    else:

        class _NoLength(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(seq_len, out_features)

            def forward(self, x):
                return self.fc(x.float())

        return _NoLength()


def _make_batch(batch_size=4, seq_len=8):
    D = torch.randint(0, 100, (batch_size, seq_len))
    DL = torch.tensor([seq_len] * batch_size)
    Energy = torch.randint(0, 3, (batch_size,))
    Risk = torch.randint(0, 2, (batch_size,))
    return D, DL, Energy, Risk


def _make_config(model, need_length=False, energy_model=False):
    return {
        "model": model,
        "device": torch.device("cpu"),
        "need_length": need_length,
        "energy_model": energy_model,
    }


# A module-level fixture or variable
@pytest.fixture
def config():
    return {
        "num_classes": None,  # Will be set per test
    }


# ── _is_better ────────────────────────────────────────────────────────────────


class TestIsBetter:
    def test_none_best_always_true(self):
        assert _is_better(0.5, None, "min") is True
        assert _is_better(0.5, None, "max") is True

    def test_min_mode_lower_is_better(self):
        assert _is_better(0.3, 0.5, "min") is True
        assert _is_better(0.7, 0.5, "min") is False

    def test_max_mode_higher_is_better(self):
        assert _is_better(0.8, 0.5, "max") is True
        assert _is_better(0.2, 0.5, "max") is False

    def test_equal_values_not_better(self):
        assert _is_better(0.5, 0.5, "min") is False
        assert _is_better(0.5, 0.5, "max") is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _is_better(0.5, 0.3, "average")


# ── _serialise_value ──────────────────────────────────────────────────────────


class TestSerialiseValue:
    def test_primitives_pass_through(self):
        assert _serialise_value(1) == 1
        assert _serialise_value(1.5) == 1.5
        assert _serialise_value("foo") == "foo"
        assert _serialise_value(True) is True
        assert _serialise_value(None) is None

    def test_list_recursed(self):
        assert _serialise_value([1, 2, 3]) == [1, 2, 3]

    def test_dict_recursed(self):
        assert _serialise_value({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_tensor_becomes_list(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _serialise_value(t)
        assert isinstance(result, list)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_unknown_type_becomes_repr(self):
        class Foo:
            pass

        result = _serialise_value(Foo())
        assert isinstance(result, str)


# ── _compute_classification_metrics ──────────────────────────────────────────



@pytest.fixture
def config():
    return {
        "num_classes": 3,
        "class_dict": {},
        "threshold": 0.80,
    }

class TestComputeClassificationMetrics:
    def test_perfect_predictions(self, config):
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        config["num_classes"] = 3
        metrics = _compute_classification_metrics(y, y, config)
        assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["f1_macro"] == pytest.approx(1.0, abs=1e-5)

    def test_all_wrong_accuracy_zero(self, config):
        y_true = torch.tensor([0, 0, 0])
        y_pred = torch.tensor([1, 1, 1])
        config["num_classes"] = 2
        metrics = _compute_classification_metrics(y_true, y_pred, config)
        assert metrics["accuracy"] == pytest.approx(0.0, abs=1e-5)

    def test_returns_expected_keys(self, config):
        y = torch.tensor([0, 1])
        config["num_classes"] = None
        metrics = _compute_classification_metrics(y, y, config)
        expected = {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "class_metrics",
            "confusion_matrix",
        }
        assert set(metrics.keys()) == expected

    def test_empty_input_returns_zeros(self, config):
        config["num_classes"] = None
        metrics = _compute_classification_metrics(torch.tensor([]), torch.tensor([]), config)
        numeric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "precision_weighted", "recall_weighted", "f1_weighted"]
        assert all(metrics[k] == 0.0 for k in numeric_keys)
        assert metrics["class_metrics"] == {}
        assert metrics["confusion_matrix"] == []

    def test_infers_num_classes(self, config):
        y = torch.tensor([0, 1, 2])
        config["num_classes"] = None
        metrics = _compute_classification_metrics(y, y, config)  # num_classes not passed
        assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-5)


# ── _unpack_batch ─────────────────────────────────────────────────────────────


class TestUnpackBatch:
    def test_without_length_returns_logits_and_risk(self):
        model = _make_model(use_length=False)
        config = _make_config(model, need_length=False, energy_model=False)
        batch = _make_batch()
        logits, targets = _unpack_batch(batch, config)
        assert logits.shape == (4, 3)
        assert torch.equal(targets, batch[3])  # Risk

    def test_with_length_passes_dl_to_model(self):
        model = _make_model(use_length=True)
        config = _make_config(model, need_length=True, energy_model=False)
        batch = _make_batch()
        logits, targets = _unpack_batch(batch, config)
        assert logits.shape == (4, 3)

    def test_energy_model_flag_selects_energy_as_target(self):
        model = _make_model(use_length=False)
        config = _make_config(model, need_length=False, energy_model=True)
        batch = _make_batch()
        _, targets = _unpack_batch(batch, config)
        assert torch.equal(targets, batch[2])  # Energy


# ── _get_learning_rates ───────────────────────────────────────────────────────


class TestGetLearningRates:
    def test_returns_list_of_lrs(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        config = {"optimiser": opt}
        lrs = _get_learning_rates(config)
        assert isinstance(lrs, list)
        assert lrs[0] == pytest.approx(1e-3)

    def test_multiple_param_groups(self):
        model = _make_model()
        opt = optim.SGD(
            [
                {"params": model.parameters(), "lr": 0.01},
            ]
        )
        config = {"optimiser": opt}
        lrs = _get_learning_rates(config)
        assert len(lrs) == 1
        assert lrs[0] == pytest.approx(0.01)

# ── get_loss_function ─────────────────────────────────────────────────────────


class TestGetLossFunction:
    def test_cross_entropy_returns_correct_type(self):
        criterion = get_loss_function("cross_entropy")
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_focal_returns_correct_type(self):
        criterion = get_loss_function("focal")
        assert isinstance(criterion, FocalLoss)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("unknown")

    def test_cross_entropy_with_weights(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        criterion = get_loss_function("cross_entropy", weight=weights)
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_focal_forward_runs(self):
        criterion = get_loss_function("focal")
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = criterion(logits, targets)
        assert loss.item() > 0

# ── imbalance ─────────────────────────────────────────────────────────────────

class TestComputeClassWeights:
    def test_returns_correct_num_classes(self):
        labels = torch.tensor([0, 0, 0, 1, 2])
        weights = compute_class_weights(labels, num_classes=3)
        assert len(weights) == 3

    def test_minority_class_gets_higher_weight(self):
        labels = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_infers_num_classes(self):
        labels = torch.tensor([0, 1, 2])
        weights = compute_class_weights(labels)
        assert len(weights) == 3


class TestMakeWeightedSampler:
    def test_returns_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        labels = torch.tensor([0, 0, 0, 1, 2])
        sampler = make_weighted_sampler(labels, num_classes=3)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_sampler_length_matches_labels(self):
        from torch.utils.data import WeightedRandomSampler
        labels = torch.tensor([0, 0, 1, 1, 2])
        sampler = make_weighted_sampler(labels, num_classes=3)
        assert len(sampler) == len(labels)