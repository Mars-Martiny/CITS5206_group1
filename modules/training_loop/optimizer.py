"""Optimizer utilities for the shared training loop."""

from __future__ import annotations

from typing import Any

import torch


def normalise_optimizer_config(
    optimiser: Any = None,
    optimiser_args: dict | None = None,
) -> dict:
    """Normalise optimizer inputs into a serialisable config dictionary.

    Backward-compatible behaviour:
    - optimiser=None uses the historical default Adam.
    - optimiser=str uses that optimizer name with optimiser_args.
    - optimiser=dict uses explicit optimizer configuration.
    - optimiser object is treated as a custom optimizer object.

    :param optimiser: Optimizer configuration, name, or object.
    :type optimiser: Any
    :param optimiser_args: Keyword arguments passed to the optimizer constructor.
    :type optimiser_args: dict | None
    :returns: Normalised optimizer configuration dictionary.
    :rtype: dict
    """
    optimiser_args = dict(optimiser_args or {})

    if optimiser is None:
        return {
            "name": "Adam",
            "lr": optimiser_args.pop("lr", 1e-3),
            **optimiser_args,
        }

    if isinstance(optimiser, str):
        return {
            "name": optimiser,
            **optimiser_args,
        }

    if isinstance(optimiser, dict):
        config = dict(optimiser)
        config.update(optimiser_args)
        config.setdefault("name", "Adam")
        config.setdefault("lr", 1e-3)
        return config

    # Backward compatibility for existing code that passes an optimizer object.
    return {
        "name": optimiser.__class__.__name__,
        "custom_object": True,
    }


def create_optimizer(
    parameters,
    optimizer_config: dict | None,
    optimizer_object=None,
):
    """Create a PyTorch optimizer from an optimizer config dictionary.

    :param parameters: Model parameters to optimise.
    :type parameters: Iterable[torch.nn.Parameter]
    :param optimizer_config: Configuration dictionary specifying optimizer type and parameters.
    :type optimizer_config: dict | None
    :param optimizer_object: Optional custom optimizer object to use if 'custom_object' is True in the config.
    :type optimizer_object: Any
    :returns: A PyTorch optimizer instance.
    :rtype: torch.optim.Optimizer
    """
    if not optimizer_config:
        optimizer_config = {
            "name": "Adam",
            "lr": 1e-3,
        }

    name = optimizer_config.get("name", "Adam")

    if optimizer_config.get("custom_object"):
        return optimizer_object

    config = dict(optimizer_config)
    config.pop("name", None)
    config.pop("custom_object", None)

    if name == "Adam":
        return torch.optim.Adam(parameters, **config)

    if name == "AdamW":
        return torch.optim.AdamW(parameters, **config)

    if name == "SGD":
        return torch.optim.SGD(parameters, **config)

    if name == "RMSprop":
        return torch.optim.RMSprop(parameters, **config)

    if name == "Adagrad":
        return torch.optim.Adagrad(parameters, **config)

    raise ValueError(f"Unsupported optimizer: {name}")