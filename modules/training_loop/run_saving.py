
"""RUN SAVING UTILITIES.

Includes:
    RunSaver class: Encapsulates history management, metrics appending, artifact saving, and plotting.
        _initialise_history: Initializes the history dictionary to store training and validation metrics.
        append_metrics: Appends the metrics for the current epoch to the history dictionary.
        save_artifacts: Saves the best model state dict and run summary to disk.

"""

import json
import math
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .utility import _serialise_value


class RunSaver:
    """Encapsulate training history, artifact saving, and metric plotting."""

    def __init__(self):
        """Initialize RunSaver with an empty history dictionary."""
        self.history = self._initialise_history()

    # HISTORY DICTIONARY // Create a new history dictionary with empty lists for each metric
    def _initialise_history(self):
        return {
            "training": {
                "train": {
                    "loss": [],
                    "accuracy": [],
                    "precision_macro": [],
                    "recall_macro": [],
                    "f1_macro": [],
                    "precision_weighted": [],
                    "recall_weighted": [],
                    "f1_weighted": [],
                    "class_metrics": [],
                    "confusion_matrix": [],
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
                    "class_metrics": [],
                    "confusion_matrix": [],
                },
                "epoch_time_sec": [],
            },
            "test": {
                "loss": [],
                "accuracy": [],
                "precision_macro": [],
                "recall_macro": [],
                "f1_macro": [],
                "precision_weighted": [],
                "recall_weighted": [],
                "f1_weighted": [],
                "class_metrics": [],
                "confusion_matrix": [],
                "auto_classification_rate": [],
                "fatal_flag_count": [],
                "fatal_flag_rate": [],
                "meets_requirement": [],
                "threshold_used": [],
                # Client requirement results (populated when config["requirements"] is set)
                "confidence_high_rate": [],
                "confidence_medium_rate": [],
                "confidence_low_rate": [],
                "req_high_confidence_met": [],
                "fatal_accuracy": [],
                "req_fatal_accuracy_met": [],
                "per_class_requirements": [],
                "req_all_f1_targets_met": [],
            },
        }

    # CREATE DIRECTORY // Create a directory for saving run artifacts based on the current timestamp
    def create_directory(self, config):
        """Create a directory for saving run artifacts."""
        parent_dir = Path(config["parent_dir"])
        save_dir = parent_dir / f"{config['timestamp']}_{config['save_name']}"
        save_dir.mkdir(parents=True, exist_ok=True)
        config["save_dir"] = save_dir  # Add save_dir to config for later use
        return save_dir

    # APPEND METRICS TO HISTORY // Append the metrics for the current epoch to the history dictionary
    def append_metrics(self, section, metrics, training=True):
        """Append the metrics for the current epoch to the history dictionary."""
        # Specify whether to append to the 'training' or 'test' section of the history dictionary
        if training:
            history_section = self.history["training"][section]
        else:
            history_section = self.history[section]
        
        # Append each metric to the corresponding list in the history dictionary
        for key in history_section.keys():
            if key in metrics:
                history_section[key].append(metrics[key])

    # SAVE RUN ARTIFACTS // Save the best model state dict and run summary to disk
    def save_artifacts(self, config, run_summary):
        """Save the best model state dict and run summary JSON to disk."""
        model_path = config["save_dir"] / f"{config['save_name']}_model.pt"
        history_path = config["save_dir"] / f"{config['save_name']}_history.pt"
        summary_path = config["save_dir"] / f"{config['save_name']}_run_summary.json"

        torch.save(run_summary["best_model_state_dict"], model_path)
        torch.save(run_summary["history"], history_path)

        serialisable_config = {
            k: _serialise_value(v)
            for k, v in config.items()
            if k
            not in {
                "model",
                "train_dl",
                "valid_dl",
                "test_dl",
                "optimiser",
                "scheduler",
                "criterion",
            }
        }

        serialisable_config["metadata"] = {
            **serialisable_config.get("metadata", {}),
            "optimiser_defaults": _serialise_value(
                config["optimiser"].defaults
                if config["optimiser"] is not None
                else None
            ),
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

    # PLOT CONFUSION MATRIX // Plot a confusion matrix as a heatmap
    def plot_confusion_matrix(self, cm, save_path, class_dict=None):
        """Plot and save a confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix as a 2D list or numpy array.
            save_path: Path to save the confusion matrix plot.
            class_dict: Optional dictionary mapping class indices to class names.
        """
        cm = np.array(cm)
        num_classes = cm.shape[0]
        
        # Create class labels
        if class_dict:
            labels = [class_dict.get(i, f"class_{i}") for i in range(num_classes)]
        else:
            labels = [f"class_{i}" for i in range(num_classes)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

    # PLOT METRICS // Generate and save plots for each metric in the history
    def plot_base_metrics(self, metric, x_values, best_epoch, save_dir, save_name, y_axis_ranges, x_max, training=True):
        """Plot a single metric for training and validation over epochs."""
        plt.figure()
        if training:
            plt.plot(x_values, self.history["training"]["train"][metric], label=f"Train {metric}")
            if metric != "lr":  # Don't plot validation metrics for learning rate
                plt.plot(x_values, self.history["training"]["val"][metric], label=f"Val {metric}")
        else:
            plt.plot(x_values, self.history["test"][metric], label=f"Test {metric}")
        plt.axvline(
            x=best_epoch,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Best epoch ({best_epoch})",
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} over Epochs")
        plt.legend(loc="best")
        plt.grid()
        if y_axis_ranges.get(metric) is not None:
            plt.ylim(*y_axis_ranges[metric])
        plt.xlim(1, x_max)
        if training:        
            plot_path = Path(save_dir) / f"{save_name}_{metric}_plot.png"
        else:
            plot_path = Path(save_dir) / f"{save_name}_test_{metric}_plot.png"
        plt.savefig(plot_path)
        plt.close()


    # PLOT HISTORY // Generate and save plots for each metric in the history
    def plot_history(self, best_epoch, save_dir, save_name):
        """Generate and save per-metric and combined plots from training history."""
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
        num_epochs = len(self.history["training"]["train"]["loss"])
        x_max = ((num_epochs + 4) // 5) * 5 if num_epochs > 0 else 5
        x_values = list(range(1, num_epochs + 1))


        # plot each metric for train and val over epochs (test is a single evaluation, not a time series)
        for metric in ["loss", "accuracy", "f1_macro", "precision_weighted", "f1_weighted"]:
            self.plot_base_metrics(metric, x_values, best_epoch, save_dir, save_name, y_axis_ranges, x_max, training=True)


        # plot learning rate if available
        if self.history["training"]["train"]["lr"]:
            self.plot_base_metrics("lr", x_values, best_epoch, save_dir, save_name, y_axis_ranges, x_max, training=True)


        # Plot class-specific metrics if available
        train_class_metrics = self.history["training"]["train"].get("class_metrics", [])
        val_class_metrics = self.history["training"]["val"].get("class_metrics", [])
        
        if train_class_metrics and isinstance(train_class_metrics[0], dict):
            # Extract class names from first epoch
            class_names = list(train_class_metrics[0].keys())
            
            # For each metric (precision, recall, f1), create per-class plots
            for metric in ["precision", "recall", "f1"]:
                plt.figure(figsize=(12, 6))
                
                for class_name in class_names:
                    # Extract metric values for this class across epochs
                    train_values = [
                        epoch_metrics.get(class_name, {}).get(metric, 0.0)
                        for epoch_metrics in train_class_metrics
                    ]
                    val_values = [
                        epoch_metrics.get(class_name, {}).get(metric, 0.0)
                        for epoch_metrics in val_class_metrics
                    ]
                    
                    plt.plot(x_values, train_values, label=f"Train {class_name}", marker="o", alpha=0.7)
                    plt.plot(x_values, val_values, label=f"Val {class_name}", marker="s", linestyle="--", alpha=0.7)
                
                plt.axvline(
                    x=best_epoch,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Best epoch ({best_epoch})",
                )
                plt.xlabel("Epoch")
                plt.ylabel(metric.capitalize())
                plt.title(f"Class-wise {metric.capitalize()} over Epochs")
                plt.legend(loc="best", fontsize=8)
                plt.grid()
                plt.ylim(0.0, 1.0)
                plt.xlim(1, x_max)
                plot_path = Path(save_dir) / f"{save_name}_class_{metric}_plot.png"
                plt.savefig(plot_path, dpi=100, bbox_inches="tight")
                plt.close()


        # Get confusion matrix metric for best epoch validation and test
        val_confusion_matrices = self.history["training"]["val"].get("confusion_matrix", [])
        test_confusion_matrices = self.history["test"].get("confusion_matrix", [])

        # Plot best epoch validation confusion matrix
        if val_confusion_matrices and len(val_confusion_matrices) >= best_epoch - 1:
            best_val_cm = val_confusion_matrices[best_epoch - 1]
            val_cm_path = Path(save_dir) / f"{save_name}_val_confusion_matrix_epoch{best_epoch}.png"
            self.plot_confusion_matrix(best_val_cm, val_cm_path)
        # Plot test confusion matrix
        if test_confusion_matrices:
            test_cm = test_confusion_matrices[-1] if isinstance(test_confusion_matrices, list) else test_confusion_matrices
            test_cm_path = Path(save_dir) / f"{save_name}_test_confusion_matrix.png"
            self.plot_confusion_matrix(test_cm, test_cm_path)


        # Combined plot: accuracy, precision_macro, recall_macro, f1_macro
        plt.figure(figsize=(12, 8))

        # Warm colors for training metrics
        train_colors = {
            "accuracy": "#e74c3c",  # red
            "precision_macro": "#e67e22",  # orange
            "recall_macro": "#f39c12",  # dark orange
            "f1_macro": "#d35400",  # dark orange-red
        }
        # Cool colors for validation metrics
        val_colors = {
            "accuracy": "#3498db",  # blue
            "precision_macro": "#2980b9",  # dark blue
            "recall_macro": "#8e44ad",  # purple
            "f1_macro": "#6c3483",  # dark purple
        }

        combined_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        for metric in combined_metrics:
            plt.plot(
                x_values,
                self.history["training"]["train"][metric],
                label=f"Train {metric}",
                color=train_colors[metric],
                linewidth=2,
            )
            plt.plot(
                x_values,
                self.history["training"]["val"][metric],
                label=f"Val {metric}",
                color=val_colors[metric],
                linewidth=2,
                linestyle="--",
            )

        plt.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best epoch ({best_epoch})",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Combined Metrics over Epochs")
        plt.legend(loc="best", fontsize=9)
        plt.grid()
        plt.ylim(0.0, 1.0)
        plt.xlim(1, x_max)
        plot_path = Path(save_dir) / f"{save_name}_combined_metrics_plot.png"
        plt.savefig(plot_path)
        plt.close()