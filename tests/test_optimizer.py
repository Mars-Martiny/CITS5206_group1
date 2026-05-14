"""Tests for optimiser factory utilities."""
import torch
import torch.nn as nn

from modules.optimisation.optimiser_factory import (
    create_optimiser,
    normalise_optimiser_config,
)


class DummyModel(nn.Module):
    """Simple dummy model for optimiser tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)


def test_normalise_optimiser_config_default():
    """Test default optimiser config."""
    config = normalise_optimiser_config()

    assert config["name"] == "Adam"
    assert config["args"]["lr"] == 1e-3


def test_normalise_optimiser_config_string():
    """Test optimiser config from string."""
    config = normalise_optimiser_config(
        optimiser="SGD",
        optimiser_args={"lr": 0.1, "momentum": 0.9},
    )

    assert config["name"] == "SGD"
    assert config["args"]["lr"] == 0.1
    assert config["args"]["momentum"] == 0.9


def test_normalise_optimiser_config_dict():
    """Test optimiser config from dictionary."""
    config = normalise_optimiser_config(
        optimiser={
            "name": "AdamW",
            "args": {"lr": 0.001},
        }
    )

    assert config["name"] == "AdamW"
    assert config["args"]["lr"] == 0.001


def test_normalise_optimiser_config_custom_object():
    """Test optimiser config from custom optimiser object."""
    model = DummyModel()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    config = normalise_optimiser_config(optimiser=optimiser)

    assert config["name"] == "Adam"
    assert config["custom_object"] is True


def test_normalise_optimiser_config_invalid_input():
    """Test invalid optimiser input."""
    config = normalise_optimiser_config(optimiser=123)

    assert config["name"] is None
    assert "error" in config


def test_create_optimiser_adam():
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


def test_create_optimiser_sgd():
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


def test_create_optimiser_adamw():
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


def test_create_optimiser_rmsprop():
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


def test_create_optimiser_adagrad():
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


def test_create_optimiser_invalid_name_returns_none():
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


def test_create_optimiser_invalid_args_returns_none():
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


def test_create_optimiser_custom_object():
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


def test_create_optimiser_invalid_custom_object():
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