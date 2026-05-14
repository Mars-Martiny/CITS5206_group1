"""Learning-rate scheduler stepping utilities for the shared training loop."""

from __future__ import annotations


def step_scheduler(
    scheduler,
    scheduler_config: dict | None,
    metrics: dict | None = None,
) -> None:
    """Step the scheduler safely.
    
    This function handles when/how the scheduler is stepped.
    
    Args:
        scheduler: The learning-rate scheduler to step.
        scheduler_config: Configuration dictionary specifying the scheduler type and parameters.
        metrics: Dictionary of metrics to use for stepping the scheduler, if required.

    Returns:
        None
    """
    if scheduler is None:
        return

    scheduler_config = scheduler_config or {}
    name = scheduler_config.get("name")

    if name == "ReduceLROnPlateau":
        monitor = scheduler_config.get("monitor", "loss")

        if metrics is None or monitor not in metrics:
            raise ValueError(
                f"ReduceLROnPlateau requires monitor metric '{monitor}', "
                "but that metric was not provided."
            )

        scheduler.step(metrics[monitor])
        return

    scheduler.step()