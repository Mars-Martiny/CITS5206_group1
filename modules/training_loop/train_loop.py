"""Main training loop with early stopping and artifact saving."""

import copy
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .one_epoch import train_one_epoch
from .validation import validate
from .run_saving import RunSaver
from .utility import _safe_class_name, _serialise_value, _is_better
from .evaluate import evaluate
from ..leaderboard import log_run
from ..optimisation.scheduler import step_scheduler
from ..leaderboard import log_run, log_search_run


#  MAIN TRAINING LOOP // ensures all control variables are consistent // compatible with Dataloader-based pipelines
def train_model_loop(
    config,
):  # all top-level inputs stored in config dictionary, which is passed around to all functions that need it
    """Manage the entire training process, including early stopping.

    Args:
        config: Dictionary containing model, dataloaders, optimizer, scheduler,
            criterion, epochs, device, patience, and all other training parameters.

    Returns:
        Dictionary containing training history, best epoch, best metric value,
        best model state dict, and total training time.
    """
    run_saver = RunSaver()
    config["save_dir"] = run_saver.create_directory(config)

    patience_counter = 0
    best_metric_value = None
    best_epoch = None
    best_model_state_dict = None

    verbose = config.get("verbose", True)

    if verbose:
        print("=" * 120)
        print(f"Training the {config['model_type']} model")
        print(f"Run name: {config['run_name']}")
        print(
            f"Best model tracked by: {config['best_metric']} ({config['best_metric_mode']})"
        )
        print("=" * 120)

    training_start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(config)
        val_metrics = validate(config)

        # if config["scheduler"] is not None and not config["scheduler_step_per_batch"]:
        #     config["scheduler"].step()
        
        if config["scheduler"] is not None and not config["scheduler_step_per_batch"]:
            step_scheduler(
                scheduler=config["scheduler"],
                scheduler_config=config.get("scheduler_config"),
                metrics=val_metrics,
                )

        run_saver.history["training"]["epoch_time_sec"].append(time.time() - epoch_start_time)

        run_saver.append_metrics("train", train_metrics)
        run_saver.append_metrics("val", val_metrics)

        current_metric_value = val_metrics[config["best_metric"]]

        if verbose:
            print("-" * 120)
            print(
                f"| Epoch {epoch:03d} "
                f"| Time: {run_saver.history['training']['epoch_time_sec'][-1]:7.2f}s "
                f"| Train Loss: {train_metrics['loss']:.4f} "
                f"| Train Acc: {train_metrics['accuracy'] * 100:.2f}% "
                f"| Val Loss: {val_metrics['loss']:.4f} "
                f"| Val Acc: {val_metrics['accuracy'] * 100:.2f}% "
                f"| Val F1 Macro: {val_metrics['f1_macro']:.4f} "
                f"| Val F1 Weighted: {val_metrics['f1_weighted']:.4f} "
                f"| Best {config['best_metric']}: {current_metric_value:.4f} |"
            )
            print("-" * 120)

        # Check for improvement and update best model if needed, otherwise increment patience counter
        if _is_better(
            current_metric_value, best_metric_value, config["best_metric_mode"]
        ):
            best_metric_value = current_metric_value
            best_epoch = epoch
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(config["model"].state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

    total_train_time = time.time() - training_start_time

    if best_model_state_dict is not None:
        config["model"].load_state_dict(best_model_state_dict)

    # Final evaluation on test set using the best model
    test_metrics = evaluate(config)
    run_saver.append_metrics("test", test_metrics, training=False)

    # Prepare run_summary for saving
    run_summary = {
        "config": config,
        "history": run_saver.history,
        "best_epoch": best_epoch,
        "best_metric_name": config["best_metric"],
        "best_metric_value": best_metric_value,
        "best_model_state_dict": best_model_state_dict,
        "training_time_sec": total_train_time,
    }

    if config["save"] and best_model_state_dict is not None:
        model_path, summary_path = run_saver.save_artifacts(
            config, run_summary
        )
        run_saver.plot_history(best_epoch, config["save_dir"], config["save_name"], class_dict=config.get("class_dict"))

        if config.get("log_leaderboard", True):
            if config.get("is_hyperparameter_search", False):
                log_search_run(
                    run_summary=run_summary,
                    config=config,
                    model_path=model_path,
                    leaderboard_dir=config.get("leaderboard_dir", "leaderboard"),
                )
            else:
                log_run(
                    run_summary=run_summary,
                    config=config,
                    model_path=model_path,
                    leaderboard_dir=config.get("leaderboard_dir", "leaderboard"),
                )

        if verbose:
            print(f"Run saved to: {config['save_dir']}")
            print(f"Total training time: {total_train_time:.4f}s")
            print(f"Best epoch: {best_epoch}")
            print(f"Best {config['best_metric']}: {best_metric_value:.6f}")
            print(f"Model saved to: {model_path}")
            print(f"Run summary saved to: {summary_path}")

    return run_summary
