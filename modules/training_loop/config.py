"""Training configuration builder for the main training loop."""

import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from .utility import _safe_class_name, _normalise_class_dict
from modules.optimisation import normalise_scheduler_config, create_scheduler, get_loss_function, make_weighted_sampler, normalise_optimiser_config, create_optimiser


# CONFIG AND UTILITY FUNCTIONS FOR TRAINING LOOP
def _build_train_config(
    model,
    energy_model,
    model_type,
    need_length,
    #
    optimiser,
    optimiser_args,
    #
    scheduler,
    scheduler_step_per_batch,
    #
    criterion_type,
    criterion_weights,
    criterion_args,
    #
    train_dl,
    valid_dl,
    test_dl,
    use_weighted_sampler,
    train_labels,
    #
    epochs,
    patience,
    num_classes,
    class_dict,
    clip_grad_max_norm,
    #
    best_metric,
    best_metric_mode,
    #
    threshold,
    temperature,
    use_temperature,
    #
    parameters,
    device,
    #
    compute_train_metrics,      # unused currently
    save,
    parent_dir,
    run_name,
    #
    extra_config,
    requirements,
):
    """Build the training configuration dictionary."""
    # Generate a timestamp for unique run identification and directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Optimiser
    optimiser_config = normalise_optimiser_config(
        optimiser=optimiser,
        optimiser_args=optimiser_args,
    )
    
    optimiser_object = optimiser
    
    optimiser = create_optimiser(
        parameters=model.parameters(),
        optimiser_config=optimiser_config,
        optimiser_object=optimiser_object,
    )


    if use_weighted_sampler and train_labels is not None:
        sampler = make_weighted_sampler(train_labels, num_classes=num_classes)
        train_dl = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=train_dl.batch_size,
            sampler=sampler,
        )
    
    scheduler_config = normalise_scheduler_config(
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
    )
    
    scheduler = create_scheduler(
        optimiser=optimiser,
        scheduler_config=scheduler_config,
        scheduler_object=scheduler if scheduler_config.get("custom_object") else None,
    )
    
    scheduler_step_per_batch = scheduler_config.get(
        "step_per_batch",
        scheduler_step_per_batch,
    )

    # LOSS FUNCTION // Get the loss function based on the specified type and weights
    criterion = get_loss_function(
        criterion_type=criterion_type,
        weight=criterion_weights,
        criterion_args=criterion_args,
        device=device,
    )

    if best_metric not in {
        "loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    }:
        raise ValueError(
            f"Invalid best_metric: {best_metric}. Must be one of 'loss', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted'."
        )

    if best_metric_mode is None:
        # Metrics containing "loss" are minimised; everything else maximised.
        best_metric_mode = "min" if "loss" in best_metric.lower() else "max"

    # Derive save_name from run_name or model_type
    if run_name:
        save_name = run_name.lower().replace(" ", "_")
    else:
        save_name = model_type.lower().replace(" ", "_")
        run_name = f"{save_name}_run_{timestamp}"

    # Normalise class_dict to ensure keys are integers and values are strings
    class_dict = _normalise_class_dict(class_dict)

    # BUILD CONFIG DICTIONARY // Build the configuration dictionary with all training parameters and metadata
    config = {
        "model": model,
        "energy_model": energy_model,
        "model_type": model_type,
        "need_length": need_length,
        #
        "optimiser": optimiser,
        #
        "scheduler": scheduler,
        "scheduler_config": scheduler_config,
        "scheduler_step_per_batch": scheduler_step_per_batch,
        #
        "criterion": criterion,
        "criterion_type": criterion_type,
        #
        "train_dl": train_dl,
        "valid_dl": valid_dl,
        "test_dl": test_dl,
        "use_weighted_sampler": use_weighted_sampler,
        #
        "epochs": epochs,
        "patience": patience,
        "num_classes": num_classes,
        "class_dict": class_dict,
        "clip_grad_max_norm": clip_grad_max_norm,
        #
        "compute_train_metrics": compute_train_metrics,
        "save": save,
        "parent_dir": parent_dir,
        "save_name": save_name,
        "run_name": run_name,
        #
        "best_metric": best_metric,
        "best_metric_mode": best_metric_mode,
        #
        "threshold": threshold,
        "temperature": temperature,
        "use_temperature": use_temperature,
        #
        "parameters": parameters,
        "device": device,
        "requirements": requirements,
        #
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": timestamp,
        # Useful run metadata
        "metadata": {
            "model_class": _safe_class_name(model),
            "optimiser_class": _safe_class_name(optimiser),
            "scheduler_class": _safe_class_name(scheduler),
            "criterion_class": _safe_class_name(criterion),
            "device": str(device),
        },
    }

    if extra_config:
        config.update(extra_config)

    return config
