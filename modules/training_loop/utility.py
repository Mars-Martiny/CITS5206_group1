"""Utility functions for the training loop.

Includes:
    _safe_class_name: 
            Safely get the class name of an object, returning None if the object is None
    _serialise_value: 
            Convert various types of values (including tensors) to a format that can be easily saved in JSON or similar formats
    _unpack_batch: 
            Unpack batches from the dataloader and prepare them for model input.
    _get_learning_rates: 
            Get the learning rates from the optimizer's parameter groups
    _is_better: 
            Compare the current metric value to the best metric value based on the specified mode (min or max)
    _normalise_class_dict:
            Convert a class dictionary with string index keys to integer index keys.
"""

import torch
import pickle
from typing import Any
from pathlib import Path
from datetime import datetime

# GET CLASS NAME // Safely get the class name of an object, returning None if the object is None
def _safe_class_name(obj):
    return obj.__class__.__name__ if obj is not None else None


# CONVERT METRICS TO SERIALISABLE FORMAT // Convert various types of values (including tensors) to a format that can be easily saved in JSON or similar formats
def _serialise_value(value):
    """Best-effort conversion for config/checkpoint metadata.

    Keeps tensors/modules/optimisers from breaking JSON-style storage.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialise_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise_value(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return repr(value)


# DATALOADER BATCH UNPACKING // unpack batches from the dataloader and prepare them for model input.
def _unpack_batch(batch, config):
    """Unpack a dataloader batch and move tensors to the configured device.

    Supports both sequence batches and BERT-style dictionary batches.

    Sequence batches use the structure ``(D, DL, Energy, Risk)``.
    BERT batches use keys such as ``input_ids``, ``attention_mask``,
    optional ``token_type_ids``, and ``label``.
    """
    device = config["device"]

    if isinstance(batch, dict):
        targets = batch["label"].to(device)

        if "attention_mask" not in batch:
            # TF-IDF style: single feature tensor, no attention mask
            logits = config["model"](batch["input_ids"].to(device))
            return logits, targets

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = config["model"](
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return logits, targets

    need_length = config["need_length"]
    energy_model = config["energy_model"]

    if need_length:
        D, DL, Energy, Risk = batch
        D = D.to(device)
        DL = DL.to(device)
        Energy = Energy.to(device)
        Risk = Risk.to(device)
        logits = config["model"](D, DL)
    else:
        D, _, Energy, Risk = batch
        D = D.to(device)
        Energy = Energy.to(device)
        Risk = Risk.to(device)
        logits = config["model"](D)

    targets = Energy if energy_model else Risk
    return logits, targets


# GET OPTIMISER LEARNING RATES // Get the learning rates from the optimizer's parameter groups
def _get_learning_rates(config):
    """Return the learning rate for each optimizer parameter group.

    Args:
        config: Training configuration dict containing the 'optimiser' key.

    Returns:
        List of learning rates, one per parameter group.
    """
    optimiser = config["optimiser"]

    return [group["lr"] for group in optimiser.param_groups]


# BEST MODEL COMPARISON // Compare the current metric value to the best metric value based on the specified mode (min or max)
def _is_better(current, best, mode):
    """Return True if current metric is better than best according to mode.

    Args:
        current: The current metric value.
        best: The best metric value so far, or None if not yet set.
        mode: 'min' if lower is better, 'max' if higher is better.

    Returns:
        True if current is better than best, False otherwise.
    """
    if best is None:
        return True

    if mode not in {"min", "max"}:
        raise ValueError(f"Invalid mode: {mode}. Expected 'min' or 'max'.")
    else:
        if mode == "min":
            return current < best
        if mode == "max":
            return current > best
        return False


# NORMALISE DICT // ensure class_dict shape to be {int : str} (i.e. class index -> class name) if it is currently {str : str}
def _normalise_class_dict(class_dict: dict[str, str]) -> dict[int, str]:
    """Convert a class dictionary with string index keys to integer index keys.

    Example:
        {"0": "Low", "1": "Medium"} -> {0: "Low", 1: "Medium"}

    Args:
        class_dict: Dictionary mapping string class indexes to class names.

    Returns:
        Dictionary mapping integer class indexes to class names.

    Raises:
        ValueError: If any key cannot be converted to an integer.
    """
    normalised: dict[int, str] = {}

    for key, value in class_dict.items():
        try:
            int_key = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class_dict key {key!r} cannot be converted to int") from exc

        normalised[int_key] = value

    return normalised


# DISPLAY NAME RESOLUTION // Resolve a display name from int-keyed or str-keyed class dictionaries.
def _class_name_for_index(class_dict, class_idx):
    """Resolve a display name from int-keyed or str-keyed class dictionaries.
    
    Args:
        class_dict: Dictionary mapping class indices (int or str) to class names.
        class_idx: The integer index of the class for which to retrieve the name.
    
    Returns:
        The class name corresponding to the given class index, or a default name if not found.
    """
    class_dict = class_dict or {}

    if class_idx in class_dict:
        return str(class_dict[class_idx])

    str_idx = str(class_idx)
    if str_idx in class_dict:
        return str(class_dict[str_idx])

    normalised = _normalise_class_dict(class_dict)
    return str(normalised.get(class_idx, f"class_{class_idx}"))


# GET CLASS NAMES IN ORDER // Return class names in class-index order for plotting labels in run_saving.py
def _ordered_class_names(class_dict, num_classes):
    """Return class names in class-index order for plotting labels.
    
    Args:
        class_dict: Dictionary mapping class indices (int or str) to class names.
        num_classes: Total number of classes, used to determine the range of class indices.
    
    Returns:
        A list of class names ordered by their class index, using the provided class_dict for name resolution.
    """
    return [_class_name_for_index(class_dict, idx) for idx in range(num_classes)]


# INFER NUMBER OF CLASSES // Infer the number of classes for class-wise plots
def _infer_num_classes_from_history(class_metrics_history, class_dict=None):
    """Infer the number of classes for class-wise plots.
    
    Args:
        class_metrics_history: A list of dictionaries containing class metrics for each epoch.
        class_dict: Optional dictionary mapping class indices to class names.
    
    Returns:
        The inferred number of classes based on the class_dict or class_metrics_history.
    """
    normalised = _normalise_class_dict(class_dict)

    int_keys = [k for k in normalised if isinstance(k, int)]
    if int_keys:
        return max(int_keys) + 1

    if class_metrics_history and isinstance(class_metrics_history[0], dict):
        return len(class_metrics_history[0])

    return 0


# EXTRACT CLASS METRICS // Extract class metrics from epoch metrics using current and legacy key names.
def _get_class_metric_value(epoch_metrics, class_idx, class_name, metric):
    """Read a class metric using current and legacy history key names.

    Current histories use the real class name, e.g. "Single Fatality".
    Older/broken histories may use "class_0" because class_dict lookup failed.
    This helper lets the plots display class_dict names while still reading
    metric values from those older histories.

    Args:
        epoch_metrics: Dictionary of metrics for the current epoch, which may contain class metrics under various key formats.
        class_idx: The integer index of the class for which to retrieve the metric.
        class_name: The display name of the class for which to retrieve the metric.
        metric: The specific metric name to retrieve (e.g., "f1", "precision", "recall").
    
    Returns:
        The value of the specified metric for the given class, or 0.0 if not found or if value cannot be converted to float.
    """
    if not isinstance(epoch_metrics, dict):
        return 0.0

    candidate_keys = (
        class_name,
        f"class_{class_idx}",
        class_idx,
        str(class_idx),
    )

    for key in candidate_keys:
        class_block = epoch_metrics.get(key)
        if isinstance(class_block, dict) and metric in class_block:
            try:
                return float(class_block[metric])
            except (TypeError, ValueError):
                return 0.0

    return 0.0