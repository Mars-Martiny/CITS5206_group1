"""Learning-rate scheduler factory utilities."""

from __future__ import annotations

from typing import Any

import torch


def normalise_scheduler_config(
    scheduler: Any = None,
    scheduler_step_per_batch: bool = False,
    best_metric: str = "loss",
    best_metric_mode: str | None = None,
) -> dict:
    """Normalise scheduler inputs into a serialisable config dictionary.

    Arguments:
        scheduler: Scheduler configuration or object. Backward-compatible behaviour:
            - scheduler=False disables scheduling.
            - scheduler=None uses the historical default StepLR.
            - scheduler=dict uses explicit scheduler configuration.
            - scheduler object is treated as a custom scheduler object.
        scheduler_step_per_batch: Whether to step the scheduler every batch instead of every epoch.
        best_metric: The metric to monitor for ReduceLROnPlateau schedulers.
        best_metric_mode: Whether to minimize or maximize the monitored metric for ReduceLROnPlateAU schedulers. If None, this is inferred from the metric name.

    Returns:
        dict: Normalised scheduler configuration dictionary.
    """
    if best_metric_mode is None:
        best_metric_mode = "min" if "loss" in best_metric.lower() else "max"

    if scheduler is False:
        return {
            "name": None,
            "step_per_batch": False,
        }

    if scheduler is None:
        return {
            "name": "StepLR",
            "step_size": 1,
            "gamma": 0.95,
            "step_per_batch": scheduler_step_per_batch,
        }

    if isinstance(scheduler, dict):
        config = dict(scheduler)
        config.setdefault("step_per_batch", scheduler_step_per_batch)

        if config.get("name") == "ReduceLROnPlateau":
            config.setdefault("monitor", best_metric)
            config.setdefault("mode", best_metric_mode)

        return config

    return {
        "name": scheduler.__class__.__name__,
        "step_per_batch": scheduler_step_per_batch,
        "custom_object": True,
    }


def create_scheduler(optimiser, scheduler_config: dict | None, scheduler_object=None):
    """Create a PyTorch scheduler from a scheduler config dictionary.
    
    Supported schedulers:
        - StepLR
        - ExponentialLR
        - CosineAnnealingLR
        - ReduceLROnPlateau
        - SequentialLR with Linear warmup followed by CosineAnnealingLR
    
    Arguments:
        optimiser: The PyTorch optimiser to attach the scheduler to.
        scheduler_config: A dictionary containing the scheduler configuration. See
            :func:`~modules.optimisation.scheduler_factory.normalise_scheduler_config` for expected keys.
        scheduler_object: If passing a custom scheduler object, this should be the instantiated scheduler to return.

    Returns:
        A PyTorch learning rate scheduler instance, or None if scheduling is disabled.
    """
    if not scheduler_config:
        return None

    name = scheduler_config.get("name")

    if name is None or str(name).lower() in {"none", "false", "disabled"}:
        return None

    if scheduler_config.get("custom_object"):
        return scheduler_object

    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size=scheduler_config.get("step_size", 1),
            gamma=scheduler_config.get("gamma", 0.95),
        )

    if name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimiser,
            gamma=scheduler_config.get("gamma", 0.95),
        )

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=scheduler_config.get("T_max", 10),
            eta_min=scheduler_config.get("eta_min", 0.0),
        )

    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 2),
            min_lr=scheduler_config.get("min_lr", 0.0),
        )

    if name == "SequentialLR":
        warmup_iters = scheduler_config.get("warmup_iters", 1)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimiser,
            start_factor=scheduler_config.get("start_factor", 1e-3),
            end_factor=scheduler_config.get("end_factor", 1.0),
            total_iters=warmup_iters,
        )

        after_scheduler_name = scheduler_config.get(
            "after_scheduler",
            "CosineAnnealingLR",
        )

        if after_scheduler_name != "CosineAnnealingLR":
            raise ValueError(
                "SequentialLR currently only supports CosineAnnealingLR "
                "as the after_scheduler."
            )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=scheduler_config.get("T_max", 10),
            eta_min=scheduler_config.get("eta_min", 0.0),
        )

        return torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iters],
        )

    raise ValueError(f"Unsupported scheduler: {name}")