import pytest
import torch.nn as nn
import torch.optim as optim

from modules.optimisation.scheduler_factory import (
    create_scheduler,
    normalise_scheduler_config,
)
from modules.optimisation.scheduler import step_scheduler


def _make_model(seq_len=8, out_features=3):
    class _NoLength(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(seq_len, out_features)

        def forward(self, x):
            return self.fc(x.float())

    return _NoLength()


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