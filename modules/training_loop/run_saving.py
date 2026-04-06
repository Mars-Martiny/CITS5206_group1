import json
import math
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from .utility import _serialise_value

"""
RUN SAVING UTILITIES 
Includes:
    RunSaver class: Encapsulates history management, metrics appending, artifact saving, and plotting.
        _initialise_history: Initializes the history dictionary to store training and validation metrics.
        append_metrics: Appends the metrics for the current epoch to the history dictionary.
        save_artifacts: Saves the best model state dict and run summary to disk.
"""

class RunSaver:
    def __init__(self):
        self.history = self._initialise_history()


    # HISTORY DICTIONARY // Create a new history dictionary with empty lists for each metric
    def _initialise_history(self):
        return {
            "train": {
                "loss": [],
                "accuracy": [],
                "precision_macro": [],
                "recall_macro": [],
                "f1_macro": [],
                "precision_weighted": [],
                "recall_weighted": [],
                "f1_weighted": [],
                "lr": [],
            },
            "val": {
                "loss": [],
                "accuracy": [],
                "precision_macro": [],
                "recall_macro": [],
                "f1_macro": [],
                "precision_weighted": [],
                "recall_weighted": [],
                "f1_weighted": [],
            },
            "epoch_time_sec": [],
        }

    # CREATE DIRECTORY // Create a directory for saving run artifacts based on the current timestamp
    def create_directory(self, config, timestamp):
        """Create a directory for saving run artifacts."""
        parent_dir = Path(config["save_dir"])
        save_dir = parent_dir / f"{timestamp}_{config['save_name']}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir


    # APPEND METRICS TO HISTORY // Append the metrics for the current epoch to the history dictionary
    def append_metrics(self, section, metrics):
        history_section = self.history[section]
        for key in history_section.keys():
            if key in metrics:
                history_section[key].append(metrics[key])

    # SAVE RUN ARTIFACTS // Save the best model state dict and run summary to disk
    def save_artifacts(self, config, run_summary, save_dir):
        model_path = save_dir / f"{config['save_name']}_model.pt"
        history_path = save_dir / f"{config['save_name']}_history.pt"
        summary_path = save_dir / f"{config['save_name']}_run_summary.json"

        torch.save(run_summary["best_model_state_dict"], model_path)
        torch.save(run_summary["history"], history_path)

        serialisable_config = {
            k: _serialise_value(v)
            for k, v in config.items()
            if k not in {"model", "train_dl", "valid_dl", "optimiser", "scheduler", "criterion"}
        }
        serialisable_config["metadata"] = {
            **serialisable_config.get("metadata", {}),
            "optimiser_defaults": _serialise_value(config["optimiser"].defaults if config["optimiser"] is not None else None),
        }

        serialisable_summary = {
            "config": serialisable_config,
            "best_epoch": run_summary["best_epoch"],
            "best_metric_name": run_summary["best_metric_name"],
            "best_metric_value": run_summary["best_metric_value"],
            "training_time_sec": run_summary["training_time_sec"],
            "history": _serialise_value(run_summary["history"]),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(serialisable_summary, f, indent=2)


        return str(model_path), str(summary_path)


    # PLOT HISTORY // Generate and save plots for each metric in the history
    def plot_history(self, best_epoch, save_dir, save_name):
        # Define standard y-axis ranges for metrics with consistent bounds
        y_axis_ranges = {
            "loss": None,
            "accuracy": (0.0, 1.0),
            "precision_macro": (0.0, 1.0),
            "recall_macro": (0.0, 1.0),
            "f1_macro": (0.0, 1.0),
            "precision_weighted": (0.0, 1.0),
            "recall_weighted": (0.0, 1.0),
            "f1_weighted": (0.0, 1.0),
            "lr": None,
        }

        # Derive a shared x-axis max rounded up to the next multiple of 5
        num_epochs = len(self.history["train"]["loss"])
        x_max = ((num_epochs + 4) // 5) * 5 if num_epochs > 0 else 5
        x_values = list(range(1, num_epochs + 1))


        # plot metrics
        metrics = self.history["val"].keys()
        for metric in metrics:
            plt.figure()
            plt.plot(x_values, self.history["train"][metric], label=f"Train {metric}")
            plt.plot(x_values, self.history["val"][metric], label=f"Val {metric}")
            plt.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best epoch ({best_epoch})')
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.title(f"{metric.capitalize()} over Epochs")
            plt.legend()
            plt.grid()
            if y_axis_ranges.get(metric) is not None:
                plt.ylim(*y_axis_ranges[metric])
            plt.xlim(1, x_max)
            plot_path = Path(save_dir) / f"{save_name}_{metric}_plot.png"
            plt.savefig(plot_path)
            plt.close()


        # plot learning rate if available
        if self.history["train"]["lr"]:
            plt.figure()
            plt.plot(x_values, self.history["train"]["lr"], label="Train lr")
            plt.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best epoch ({best_epoch})')
            plt.xlabel("Epoch")
            plt.ylabel("Learning rate")
            plt.title("Learning rate over Epochs")
            plt.legend()
            plt.grid()
            if y_axis_ranges.get("lr") is not None:
                plt.ylim(*y_axis_ranges["lr"])
            plt.xlim(1, x_max)
            plot_path = Path(save_dir) / f"{save_name}_lr_plot.png"
            plt.savefig(plot_path)
            plt.close()


        # Combined plot: accuracy, precision_macro, recall_macro, f1_macro
        plt.figure(figsize=(12, 8))
        
        # Warm colors for training metrics
        train_colors = {
            "accuracy": "#e74c3c",      # red
            "precision_macro": "#e67e22",  # orange
            "recall_macro": "#f39c12",     # dark orange
            "f1_macro": "#d35400",         # dark orange-red
        }
        # Cool colors for validation metrics
        val_colors = {
            "accuracy": "#3498db",       # blue
            "precision_macro": "#2980b9", # dark blue
            "recall_macro": "#8e44ad",    # purple
            "f1_macro": "#6c3483",        # dark purple
        }
        
        combined_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        
        for metric in combined_metrics:
            plt.plot(x_values, self.history["train"][metric], 
                    label=f"Train {metric}", color=train_colors[metric], linewidth=2)
            plt.plot(x_values, self.history["val"][metric], 
                    label=f"Val {metric}", color=val_colors[metric], linewidth=2, linestyle='--')
        
        plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, label=f'Best epoch ({best_epoch})')
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Combined Metrics over Epochs")
        plt.legend(loc='best', fontsize=9)
        plt.grid()
        plt.ylim(0.0, 1.0)
        plt.xlim(1, x_max)
        plot_path = Path(save_dir) / f"{save_name}_combined_metrics_plot.png"
        plt.savefig(plot_path)
        plt.close()
