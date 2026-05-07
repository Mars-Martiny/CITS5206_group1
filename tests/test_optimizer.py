"""Tests for optimizer utilities."""

import pytest
import torch

from modules.training_loop.optimizer import (
    create_optimizer,
    normalise_optimizer_config,
)


def _dummy_model():
    """Create a small model for optimizer tests."""
    return torch.nn.Linear(4, 2)


def test_normalise_optimizer_config_default_adam():
    config = normalise_optimizer_config()

    assert config["name"] == "Adam"
    assert config["lr"] == 1e-3


def test_normalise_optimizer_config_string_with_args():
    config = normalise_optimizer_config(
        optimiser="AdamW",
        optimiser_args={
            "lr": 1e-4,
            "weight_decay": 0.01,
        },
    )

    assert config["name"] == "AdamW"
    assert config["lr"] == 1e-4
    assert config["weight_decay"] == 0.01


def test_create_optimizer_adam():
    model = _dummy_model()

    optimizer = create_optimizer(
        parameters=model.parameters(),
        optimizer_config={
            "name": "Adam",
            "lr": 1e-3,
        },
    )

    assert isinstance(optimizer, torch.optim.Adam)


def test_create_optimizer_adamw():
    model = _dummy_model()

    optimizer = create_optimizer(
        parameters=model.parameters(),
        optimizer_config={
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
        },
    )

    assert isinstance(optimizer, torch.optim.AdamW)


def test_create_optimizer_sgd_with_momentum():
    model = _dummy_model()

    optimizer = create_optimizer(
        parameters=model.parameters(),
        optimizer_config={
            "name": "SGD",
            "lr": 0.01,
            "momentum": 0.9,
        },
    )

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9


def test_create_optimizer_rmsprop():
    model = _dummy_model()

    optimizer = create_optimizer(
        parameters=model.parameters(),
        optimizer_config={
            "name": "RMSprop",
            "lr": 1e-3,
        },
    )

    assert isinstance(optimizer, torch.optim.RMSprop)


def test_create_optimizer_unsupported_name_raises():
    model = _dummy_model()

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        create_optimizer(
            parameters=model.parameters(),
            optimizer_config={
                "name": "FakeOptimizer",
                "lr": 1e-3,
            },
        )