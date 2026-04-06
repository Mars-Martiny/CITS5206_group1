import time

import torch
import torch.nn as nn
import torch.optim as optim

from .utility import _safe_class_name

# CONFIG AND UTILITY FUNCTIONS FOR TRAINING LOOP
def _build_train_config(model, train_dl, valid_dl, epochs, device, patience, criterion_weights, model_type="Simple", save=True, optimiser=None, scheduler=None, criterion=None,
                  need_length=False, energy_model=False, best_metric="loss", best_metric_mode=None, clip_grad_max_norm=1.0, scheduler_step_per_batch=False,
                  save_dir="trained_models", run_name=None, num_classes=None, extra_config=None,
                 ):
    """Build the training configuration dictionary.""" 

    # DEFAULT OPTIMISER, SCHEDULER, CRITERION // Set default optimiser, scheduler, and criterion if not provided
    if optimiser is None:
        optimiser = optim.Adam(model.parameters(), lr=1e-3)

    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.95)

    if criterion is None:
        if criterion_weights is not None:
            criterion_weights = criterion_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=criterion_weights)

    if best_metric not in {"loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"}:
        raise ValueError(
            f"Invalid best_metric: {best_metric}. Must be one of 'loss', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted'."
        )

    if best_metric_mode is None:
        # Metrics containing "loss" are minimised; everything else maximised.
        best_metric_mode = "min" if "loss" in best_metric.lower() else "max"

    # Derive save_name from run_name or model_type
    if run_name:
        save_name = run_name.lower().replace(" ", "_")[:10]
    else:
        save_name = model_type.lower().replace(" ", "_")[:10]

    config = {
        "model": model,
        "train_dl": train_dl,
        "valid_dl": valid_dl,
        "epochs": epochs,
        "device": device,
        "patience": patience,
        "criterion_weights": criterion_weights,
        "criterion": criterion,
        "optimiser": optimiser,
        "scheduler": scheduler,
        "model_type": model_type,
        "save": save,
        "save_dir": save_dir,
        "save_name": save_name,
        "run_name": run_name or f"{save_name}_run_{int(time.time())}",
        "need_length": need_length,
        "energy_model": energy_model,
        "best_metric": best_metric,
        "best_metric_mode": best_metric_mode,
        "clip_grad_max_norm": clip_grad_max_norm,
        "scheduler_step_per_batch": scheduler_step_per_batch,
        "num_classes": num_classes,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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