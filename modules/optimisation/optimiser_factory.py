"""Optimiser factory for creating optimisers with different configurations."""

from __future__ import annotations

from typing import Any

import torch


SUPPORTED_OPTIMISERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "Adagrad": torch.optim.Adagrad,
}


def normalise_optimiser_config(
    optimiser: Any = None,
    optimiser_args: dict | None = None,
) -> dict:
    """Normalise optimiser inputs into a serialisable config dictionary.

    Args:
        optimiser:
            Optimiser configuration.

            Acceptable values:
            - None: use the default Adam optimiser.
            - str: one of "Adam", "AdamW", "SGD", "RMSprop", or "Adagrad".
            - dict: explicit configuration, for example:
                {"name": "Adam", "args": {"lr": 0.001}}
            - torch.optim.Optimizer: pre-created optimiser object.

        optimiser_args:
            Keyword arguments passed to the optimiser constructor.

            Acceptable examples:
            - {"lr": 0.001}
            - {"lr": 0.01, "momentum": 0.9} for SGD
            - {"lr": 0.001, "weight_decay": 0.01} for AdamW

    Returns:
        A normalised optimiser configuration dictionary.
    """
    optimiser_args = optimiser_args or {}

    if optimiser is None:
        return {
            "name": "Adam",
            "args": {"lr": optimiser_args.get("lr", 1e-3)},
        }

    if isinstance(optimiser, str):
        return {
            "name": optimiser,
            "args": optimiser_args,
        }

    if isinstance(optimiser, dict):
        config = dict(optimiser)
        config.setdefault("args", {})
        return config

    if isinstance(optimiser, torch.optim.Optimizer):
        return {
            "name": optimiser.__class__.__name__,
            "args": {},
            "custom_object": True,
        }

    return {
        "name": None,
        "args": {},
        "error": (
            "Invalid optimiser input. Expected None, a supported optimiser "
            "name, a config dictionary, or a torch.optim.Optimizer object."
        ),
    }


def create_optimiser(
    parameters,
    optimiser_config: dict | None,
    optimiser_object: torch.optim.Optimizer | None = None,
):
    """Create a PyTorch optimiser from an optimiser config dictionary.

    Args:
        parameters:
            Model parameters, usually model.parameters().

        optimiser_config:
            Normalised optimiser configuration dictionary.

            Expected format:
                {
                    "name": "Adam",
                    "args": {"lr": 0.001}
                }

            Acceptable optimiser names:
            - "Adam"
            - "AdamW"
            - "SGD"
            - "RMSprop"
            - "Adagrad"

        optimiser_object:
            Optional pre-created torch.optim.Optimizer object.

    Returns:
        A torch.optim.Optimizer object, or None if the configuration is invalid.
    """
    if optimiser_config is None:
        optimiser_config = {
            "name": "Adam",
            "args": {"lr": 1e-3},
        }

    if optimiser_config.get("error"):
        print(f"[Optimiser Factory] {optimiser_config['error']}")
        return None

    name = optimiser_config.get("name")
    args = optimiser_config.get("args", {})

    if name is None:
        print("[Optimiser Factory] No optimiser name provided.")
        return None

    if optimiser_config.get("custom_object"):
        if isinstance(optimiser_object, torch.optim.Optimizer):
            return optimiser_object

        print(
            "[Optimiser Factory] Custom optimiser object was requested, "
            "but no valid torch.optim.Optimizer object was provided."
        )
        return None

    optimiser_class = SUPPORTED_OPTIMISERS.get(name)

    if optimiser_class is None:
        print(
            f"[Optimiser Factory] Unsupported optimiser '{name}'. "
            f"Supported values are: {list(SUPPORTED_OPTIMISERS.keys())}."
        )
        return None

    try:
        return optimiser_class(parameters, **args)
    except TypeError as exc:
        print(
            f"[Optimiser Factory] Invalid arguments for optimiser '{name}': "
            f"{args}. Original error: {exc}"
        )
        return None