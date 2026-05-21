
"""RUN SAVING UTILITIES.

Includes:
    RunSaver class: Encapsulates history management, metrics appending, artifact saving, and plotting.
        _initialise_history:    Initializes the history dictionary to store training and validation metrics.
        create_directory:       Creates a directory for saving run artifacts based on the current timestamp. 
        append_metrics:         Appends the metrics for the current epoch to the history dictionary.
        save_artifacts:         Saves the best model state dict and run summary to disk.
        plot_confusion_matrix:  Plots a confusion matrix as a heatmap.
        plot_base_metrics:      Plots a single metric for training and validation over epochs.
        plot_history:           Generates and saves plots for each metric in the history, including combined plots and class-specific metrics.

"""

import json
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from .utility import _serialise_value, _ordered_class_names, _infer_num_classes_from_history, _get_class_metric_value


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
        
        # Get class labels
        labels = _ordered_class_names(class_dict, num_classes)
        
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
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()


    # PLOT METRICS // Generate and save plots for each metric in the history
    def plot_combined_metrics(self, title, metrics, x_values, best_epoch, plot_path, range_0_1, x_max, 
                                class_name=None, class_idx=None, train_class_metrics=None, val_class_metrics=None):
        """Plot metrics listed in 'metrics' for training and validation over epochs."""
        # Colours for metrics
        metric_colors = ["#3498db", "#8e44ad", "#f39c12", "#e74c3c"]  # blue, purple, dark orange, red

        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            if class_name is not None:
                train_values = [
                        _get_class_metric_value(epoch_metrics, class_idx, class_name, metric)
                        for epoch_metrics in train_class_metrics
                    ]

                val_values = [
                    _get_class_metric_value(epoch_metrics, class_idx, class_name, metric)
                    for epoch_metrics in val_class_metrics
                ]

            else:
                train_values = self.history["training"]["train"][metric]
                if metric != "lr":  # Don't plot validation metrics for learning rate since it's not a performance metric and may have a different scale
                    val_values = self.history["training"]["val"][metric]

            plt.plot(
                x_values,
                train_values,
                label=f"Train {metric}",
                color=metric_colors[i],
                linewidth=2,
                alpha=0.7,
            )
            if metric != "lr":  # Don't plot validation metrics for learning rate since it's not a performance metric and may have a different scale
                plt.plot(
                    x_values,
                    val_values,
                    label=f"Val {metric}",
                    color=metric_colors[i],
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )

        plt.axvline(
            x=best_epoch,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Best epoch ({best_epoch})",
            alpha=0.55,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{title} over Epochs")
        plt.legend(loc="best", fontsize=8)
        plt.grid()
        if range_0_1:
            plt.ylim(0.0, 1.1)
        plt.xlim(1, x_max)
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()


    # PLOT HISTORY // Generate and save plots for each metric in the history
    def plot_history(self, best_epoch, config):
        """Generate and save per-metric and combined plots from training history."""
        # Make a "plots" subdirectory for all plots in save_dir
        save_dir = Path(config["save_dir"])          # Use the save_dir from config as parent
        plot_dir = save_dir / "plot_metrics"                
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Extract config values frequently needed for plotting
        save_name = config["save_name"]
        class_dict = config["class_dict"]

        # Derive a shared x-axis max rounded up to the next multiple of 5
        num_epochs = len(self.history["training"]["train"]["loss"])
        x_max = ((num_epochs + 4) // 5) * 5 if num_epochs > 0 else 5
        x_values = list(int(x) for x in range(1, num_epochs + 1))


        # COMBINED PLOTS // Plot multiple metrics together for easier comparison:     
        # - Macro metrics plot: accuracy, precision_macro, recall_macro, f1_macro
        macro_metrics = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
        plot_path = Path(plot_dir) / f"{save_name}_combined_macro_metrics_plot.png"
        title = "Macro metrics over Epochs"
        self.plot_combined_metrics(title, macro_metrics, x_values, best_epoch, plot_path, True, x_max)

        # - Weighted metrics plot: accuracy, precision_weighted, recall_weighted, f1_weighted
        weighted_metrics = ["precision_weighted", "recall_weighted", "f1_weighted", "accuracy"]
        plot_path = Path(plot_dir) / f"{save_name}_combined_weighted_metrics_plot.png"
        title = "Weighted metrics over Epochs"
        self.plot_combined_metrics(title, weighted_metrics, x_values, best_epoch, plot_path, True, x_max)


        # PLOT LOSS
        weighted_metrics = ["loss"]
        plot_path = Path(plot_dir) / f"{save_name}_loss_metric_plot.png"
        title = "Loss over Epochs"
        self.plot_combined_metrics(title, weighted_metrics, x_values, best_epoch, plot_path, False, x_max)


        # PLOT LEARNING RATE
        if self.history["training"]["train"]["lr"]:
            lr_metrics = ["lr"]
            plot_path = Path(plot_dir) / f"{save_name}_learning_rate_plot.png"
            title = "Learning Rate over Epochs"
            self.plot_combined_metrics(title, lr_metrics, x_values, best_epoch, plot_path, False, x_max)        


        # PLOT CLASS-SPECIFIC METRICS // if class-specific metrics available
        train_class_metrics = self.history["training"]["train"].get("class_metrics", [])
        val_class_metrics = self.history["training"]["val"].get("class_metrics", [])

        if train_class_metrics and isinstance(train_class_metrics[0], dict):
            # Make class metrics directory // there are quite a few classes and metrics, so we put them in a separate "class_metrics" subdirectory to keep things organised
            class_dir = plot_dir / "class_metrics"
            class_dir.mkdir(parents=True, exist_ok=True)

            num_classes = _infer_num_classes_from_history(train_class_metrics, class_dict)
            class_names = _ordered_class_names(class_dict, num_classes)
            
            # For each metric (precision, recall, f1), create per-class plots.
            # Legends always use class_dict names. Values are read from current
            # class-name keys, with fallback support for older "class_N" histories.
            for class_idx, class_name in enumerate(class_names):
                metrics = ["precision", "recall", "f1"]
                save_class_name = class_name.lower().replace(" ", "_").replace("/", "_")
                plot_path = Path(class_dir) / f"{save_name}_{save_class_name}_metrics_plot.png"
                title = f"{class_name} Class metrics over Epochs"

                self.plot_combined_metrics(title, metrics, x_values, best_epoch, plot_path, True, x_max, 
                                        class_name, class_idx, train_class_metrics, val_class_metrics)


        # PLOT CONFUSION MATRICES // Get confusion matrix metric for best epoch validation and test
        val_confusion_matrices = self.history["training"]["val"].get("confusion_matrix", [])
        test_confusion_matrices = self.history["test"].get("confusion_matrix", [])

        # Plot best epoch validation confusion matrix
        if val_confusion_matrices and len(val_confusion_matrices) >= best_epoch - 1:
            best_val_cm = val_confusion_matrices[best_epoch - 1]
            val_cm_path = Path(plot_dir) / f"{save_name}_confusion_matrix_val_best_epoch.png"
            self.plot_confusion_matrix(best_val_cm, val_cm_path, class_dict=class_dict)

        # Plot test confusion matrix
        if test_confusion_matrices:
            test_cm = test_confusion_matrices[-1] if isinstance(test_confusion_matrices, list) else test_confusion_matrices
            test_cm_path = Path(plot_dir) / f"{save_name}_confusion_matrix_test.png"
            self.plot_confusion_matrix(test_cm, test_cm_path, class_dict=class_dict)

