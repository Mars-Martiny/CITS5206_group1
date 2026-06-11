"""
Unit tests for modules/optimisation functions.

Covers pure/side-effect-free functions that don't require a full training run:
    - get_loss_function
    - FocalLoss
    - compute_class_weights
    - make_weighted_sampler
    - create_scheduler
    - normalise_scheduler_config
    - step_scheduler

Run with (from the project root): pytest tests/test_optimisation.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


from modules.optimisation import (
                                    get_loss_function, 
                                    FocalLoss, 
                                    make_weighted_sampler, 
                                    create_scheduler, 
                                    normalise_scheduler_config,
                                    create_optimiser,
                                    normalise_optimiser_config,
                                 )
from modules.optimisation.imbalance import compute_class_weights
from modules.optimisation.scheduler import step_scheduler



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


class DummyModel(nn.Module):
    """Simple dummy model for optimiser tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)


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


# ── Imbalance ─────────────────────────────────────────────────────────────────

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


# ── Scheduler Utilities ────────────────────────────────────────────────────────────────

class TestSchedulerUtilities:
    def test_false_scheduler_disables_scheduler(self):
        config = normalise_scheduler_config(
            scheduler=False,
            scheduler_step_per_batch=False,
            best_metric="loss",
            best_metric_mode="min",
        )
        assert config["name"] is None

    def test_none_scheduler_defaults_to_step_lr(self):
        config = normalise_scheduler_config(
            scheduler=None,
            scheduler_step_per_batch=False,
            best_metric="loss",
            best_metric_mode="min",
        )
        assert config["name"] == "StepLR"
        assert config["step_size"] == 1
        assert config["gamma"] == pytest.approx(0.95)

    def test_create_step_lr_scheduler(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "StepLR",
            "step_size": 1,
            "gamma": 0.5,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        assert isinstance(scheduler, optim.lr_scheduler.StepLR)

    def test_step_lr_updates_learning_rate(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "StepLR",
            "step_size": 1,
            "gamma": 0.5,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        opt.step()
        step_scheduler(scheduler, scheduler_config)

        assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)

    def test_create_reduce_on_plateau_scheduler(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "ReduceLROnPlateau",
            "monitor": "f1_macro",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    def test_reduce_on_plateau_requires_monitor_metric(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "ReduceLROnPlateau",
            "monitor": "f1_macro",
            "mode": "max",
        }

        scheduler = create_scheduler(opt, scheduler_config)

        with pytest.raises(ValueError):
            step_scheduler(
                scheduler=scheduler,
                scheduler_config=scheduler_config,
                metrics={"loss": 1.0},
            )

    def test_create_cosine_scheduler(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "CosineAnnealingLR",
            "T_max": 10,
            "eta_min": 1e-6,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)


# ── Optimiser Tests ─────────────────────────────────────────────────────────


class TestOptimiserUtilities:
    def test_normalise_optimiser_config_default(self):
        """Test default optimiser config."""
        config = normalise_optimiser_config()

        assert config["name"] == "Adam"
        assert config["args"]["lr"] == 1e-3


    def test_normalise_optimiser_config_string(self):
        """Test optimiser config from string."""
        config = normalise_optimiser_config(
            optimiser="SGD",
            optimiser_args={"lr": 0.1, "momentum": 0.9},
        )

        assert config["name"] == "SGD"
        assert config["args"]["lr"] == 0.1
        assert config["args"]["momentum"] == 0.9


    def test_normalise_optimiser_config_dict(self):
        """Test optimiser config from dictionary."""
        config = normalise_optimiser_config(
            optimiser={
                "name": "AdamW",
                "args": {"lr": 0.001},
            }
        )

        assert config["name"] == "AdamW"
        assert config["args"]["lr"] == 0.001


    def test_normalise_optimiser_config_custom_object(self):
        """Test optimiser config from custom optimiser object."""
        model = DummyModel()

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

        config = normalise_optimiser_config(optimiser=optimiser)

        assert config["name"] == "Adam"
        assert config["custom_object"] is True


    def test_normalise_optimiser_config_invalid_input(self):
        """Test invalid optimiser input."""
        config = normalise_optimiser_config(optimiser=123)

        assert config["name"] is None
        assert "error" in config


    def test_create_optimiser_adam(self):
        """Test Adam optimiser creation."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "Adam",
                "args": {"lr": 1e-3},
            },
        )

        assert isinstance(optimiser, torch.optim.Adam)


    def test_create_optimiser_sgd(self):
        """Test SGD optimiser creation."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "SGD",
                "args": {"lr": 0.1, "momentum": 0.9},
            },
        )

        assert isinstance(optimiser, torch.optim.SGD)


    def test_create_optimiser_adamw(self):
        """Test AdamW optimiser creation."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "AdamW",
                "args": {"lr": 1e-3},
            },
        )

        assert isinstance(optimiser, torch.optim.AdamW)


    def test_create_optimiser_rmsprop(self):
        """Test RMSprop optimiser creation."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "RMSprop",
                "args": {"lr": 1e-3},
            },
        )

        assert isinstance(optimiser, torch.optim.RMSprop)


    def test_create_optimiser_adagrad(self):
        """Test Adagrad optimiser creation."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "Adagrad",
                "args": {"lr": 1e-3},
            },
        )

        assert isinstance(optimiser, torch.optim.Adagrad)


    def test_create_optimiser_invalid_name_returns_none(self):
        """Test unsupported optimiser name."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "FakeOptimiser",
                "args": {"lr": 1e-3},
            },
        )

        assert optimiser is None


    def test_create_optimiser_invalid_args_returns_none(self):
        """Test invalid optimiser arguments."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "Adam",
                "args": {
                    "lr": 1e-3,
                    "momentum": 0.9,
                },
            },
        )

        assert optimiser is None


    def test_create_optimiser_custom_object(self):
        """Test returning custom optimiser object."""
        model = DummyModel()

        custom_optimiser = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "Adam",
                "args": {},
                "custom_object": True,
            },
            optimiser_object=custom_optimiser,
        )

        assert optimiser is custom_optimiser


    def test_create_optimiser_invalid_custom_object(self):
        """Test invalid custom optimiser object."""
        model = DummyModel()

        optimiser = create_optimiser(
            parameters=model.parameters(),
            optimiser_config={
                "name": "Adam",
                "args": {},
                "custom_object": True,
            },
            optimiser_object="not an optimiser",
        )

        assert optimiser is None