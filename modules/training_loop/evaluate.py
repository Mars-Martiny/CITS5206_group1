"""Training loop module for evaluation."""

import json
import torch
import torch.nn.functional as F
from pathlib import Path

from .metrics import _compute_classification_metrics
from .utility import _unpack_batch, _normalise_class_dict, _class_name_for_index
from .run_saving import RunSaver
from modules.inference import run_inference

FATAL_CLASSES = ["Fatal", "Single Fatality", "Multiple Fatality"]

_DEFAULT_REQUIREMENTS_PATH = Path(__file__).parents[2] / "config_requirement_check.json"


def _load_default_requirements():
    if _DEFAULT_REQUIREMENTS_PATH.exists():
        with open(_DEFAULT_REQUIREMENTS_PATH) as f:
            return json.load(f)
    return {}


def _normalise_threshold(value):
    """Convert percentage (>1) to fraction; leave fractions unchanged."""
    return value / 100 if value > 1 else value


def _fatal_class_indices(config):
    """Return integer class indices whose names match FATAL_CLASSES."""
    class_dict = _normalise_class_dict(config.get("class_dict", {}))
    return [
        idx for idx, name in class_dict.items()
        if isinstance(idx, int) and name in FATAL_CLASSES
    ]


def _check_requirements(all_targets, all_probs, all_preds, metrics, config):
    """Check whether the model meets each client performance requirement.

    Reads the ``requirements`` key from config with the shape::

        {
            "confidence_threshold": {"high": 0.80, "medium": 0.50},
            "high_threshold": 0.70,     # min fraction in high-confidence tier
            "fatal_accuracy": 0.95,     # min recall on true-fatal samples
            "f1_target": {0: 0.70, 1: 0.70, 2: 0.0}  # 0.0 = no target
        }

    Threshold values > 1 are treated as percentages and normalised to [0, 1].

    Returns:
        Flat dict of requirement results, or empty dict if ``requirements`` is None.
    """
    requirements = config.get("requirements", {})
    if requirements is None:
        return {}

    defaults = _load_default_requirements()
    requirements = {**defaults, **requirements}

    result = {}
    n = max(len(all_probs), 1)
    max_probs = all_probs.max(dim=1).values if all_probs.numel() > 0 else torch.tensor([])

    # Confidence tier distribution
    conf_thresholds = requirements.get("confidence_threshold", {})
    high_t = _normalise_threshold(conf_thresholds.get("high", 0.80))
    medium_t = _normalise_threshold(conf_thresholds.get("medium", 0.50))

    result["confidence_high_rate"] = (max_probs >= high_t).sum().item() / n
    result["confidence_medium_rate"] = (
        ((max_probs >= medium_t) & (max_probs < high_t)).sum().item() / n
    )
    result["confidence_low_rate"] = (max_probs < medium_t).sum().item() / n

    # >= X% of predictions must fall in the high-confidence tier
    high_threshold = requirements.get("high_threshold", 0.70)
    result["req_high_confidence_met"] = bool(result["confidence_high_rate"] >= high_threshold)

    # Recall on true fatal samples >= target
    if not config.get("energy_model", False):
        fatal_indices = _fatal_class_indices(config)
        if fatal_indices:
            fatal_tensor = torch.tensor(
                fatal_indices,
                dtype=all_targets.dtype,
                device=all_targets.device,
            )
            true_fatal_mask = torch.isin(all_targets, fatal_tensor)
            fatal_total = true_fatal_mask.sum().item()
            if fatal_total > 0:
                fatal_correct = (
                    (all_preds[true_fatal_mask] == all_targets[true_fatal_mask])
                    .sum()
                    .item()
                )
                fatal_accuracy = fatal_correct / fatal_total
            else:
                fatal_accuracy = None

            fatal_req = requirements.get("fatal_accuracy", 0.95)
            result["fatal_accuracy"] = fatal_accuracy
            result["req_fatal_accuracy_met"] = (
                bool(fatal_accuracy >= fatal_req) if fatal_accuracy is not None else None
            )

    # Per-class F1 >= target (target == 0.0 means no requirement for that class)
    f1_targets = requirements.get("f1_target", {})
    class_dict = config.get("class_dict", {})
    class_metrics = metrics.get("class_metrics", {})

    per_class_req = {}
    for idx_key, target_f1 in f1_targets.items():
        idx = int(idx_key)
        class_name = _class_name_for_index(class_dict, idx)

        # Normal metric output is keyed by class name. Fallback to class_N for
        # older histories produced before class_dict name resolution was fixed.
        actual_f1 = (
            class_metrics.get(class_name, {})
            .get("f1", class_metrics.get(f"class_{idx}", {}).get("f1", 0.0))
        )

        has_target = target_f1 > 0.0
        per_class_req[class_name] = {
            "f1": actual_f1,
            "target": target_f1 if has_target else None,
            "met": bool(actual_f1 >= target_f1) if has_target else None,
        }

    result["per_class_requirements"] = per_class_req
    targeted = [v["met"] for v in per_class_req.values() if v["met"] is not None]
    result["req_all_f1_targets_met"] = bool(all(targeted)) if targeted else None

    return result


def evaluate(config):
    """Evaluation function for evaluating the model on the test set.

    Extends the validation function with additional evaluation metrics
    based on project success criteria.

    Args:
        config (dict): A configuration dictionary containing the following keys:
            - model: The trained model to be evaluated.
            - test_dl: The test dataloader.
            - criterion: The loss function.
            - device: The device to run the evaluation on.
            - num_classes: The number of classes in the classification task.
            - class_dict: Dictionary mapping class indices to class names.
            - save_dir: Directory where the model was saved.
            - threshold: Confidence threshold for auto-classification. Defaults to 0.80.
            - temperature: Temperature value for scaling logits. Defaults to 1.5.
            - use_temperature: Whether to use temperature scaling. Defaults to True.
            - requirements: Client performance requirements dict with keys:
                - confidence_threshold: {"high": float, "medium": float}
                - high_threshold: min fraction in high-confidence tier (default 0.70)
                - fatal_accuracy: min recall on true fatal samples (default 0.95)
                - f1_target: {class_index: min_f1} — 0.0 means no target

    Returns:
        metrics (dict): A dictionary containing:
            - accuracy, precision_macro, recall_macro, f1_macro
            - precision_weighted, recall_weighted, f1_weighted
            - loss
            - auto_classification_rate: proportion of predictions above threshold
            - meets_requirement: whether auto_classification_rate >= 0.70
            - threshold_used: the threshold value used
            - fatal_flag_count: number of predictions flagged as fatal
            - fatal_flag_rate: proportion of all predictions that are fatal
            -> If requirements configured, also:
                - confidence_high_rate, confidence_medium_rate, confidence_low_rate
                - req_high_confidence_met
                - fatal_accuracy: recall on true fatal samples
                - req_fatal_accuracy_met
                - per_class_requirements: per-class F1 check results
                - req_all_f1_targets_met
    """
    if config["test_dl"] is None:
        return {}

    model = config["model"]
    criterion = config["criterion"]

    model.eval()

    total_loss, total_examples = 0.0, 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in config["test_dl"]:
            logits, targets = _unpack_batch(batch, config)
            loss = criterion(logits, targets)

            if config["use_temperature"]:
                probs = F.softmax(logits / config["temperature"], dim=1)
            else:
                probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_probs.append(probs.detach().cpu())

    avg_loss = total_loss / max(total_examples, 1)
    all_preds = torch.cat(all_preds) if all_preds else torch.tensor([], dtype=torch.long)
    all_targets = torch.cat(all_targets) if all_targets else torch.tensor([], dtype=torch.long)
    all_probs = torch.cat(all_probs) if all_probs else torch.empty((0, config["num_classes"]))

    # Compute base classification metrics from metrics.py
    metrics = _compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        config=config,
    )
    metrics["loss"] = avg_loss

    # Threshold analysis - auto classification rate
    max_probs = all_probs.max(dim=1).values if all_probs.numel() > 0 else torch.tensor([])
    high_confidence = (max_probs > config["threshold"]).sum().item()
    metrics["auto_classification_rate"] = high_confidence / max(total_examples, 1)
    metrics["meets_requirement"] = metrics["auto_classification_rate"] >= 0.70
    metrics["threshold_used"] = config["threshold"]

    # Fatal category flagging - proportion of all predictions that are fatal
    if not config["energy_model"]:
        fatal_indices = _fatal_class_indices(config)
        if fatal_indices:
            fatal_tensor = torch.tensor(
                fatal_indices,
                dtype=all_preds.dtype,
                device=all_preds.device,
            )
            fatal_mask = torch.isin(all_preds, fatal_tensor)
            metrics["fatal_flag_count"] = fatal_mask.sum().item()
            metrics["fatal_flag_rate"] = fatal_mask.sum().item() / max(total_examples, 1)

    # Client requirement checks
    req_results = _check_requirements(all_targets, all_probs, all_preds, metrics, config)
    metrics.update(req_results)

    return metrics
