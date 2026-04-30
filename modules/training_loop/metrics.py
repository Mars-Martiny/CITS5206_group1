"""Classification metrics computation using pure PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim


# CLASSIFICATION METRICS COMPUTATION // Compute common classification metrics (accuracy, precision, recall, F1) from true and predicted labels
# Pure PyTorch implementation to avoid adding a hard dependency on sklearn.
def _compute_classification_metrics(
    y_true,
    y_pred,
    config,
):
    """Compute classification metrics using pure PyTorch.

    Args:
        y_true: Tensor of shape (N,)
        y_pred: Tensor of shape (N,)
        config (dict): A configuration dictionary containing the following keys:
            - num_classes: optional manual class count (inferred from data if None)
            - class_dict: dict mapping class index -> class name

    Returns:
        Dictionary containing:
            - accuracy
            - macro / weighted precision, recall, f1
            - class-specific metrics
            - confusion matrix
    """
    # Notes:
    #     Class labels must be integer-encoded from 0 to num_classes-1.
    #     If y_true is empty, all metrics are returned as 0.0.

    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)

    if y_true.numel() == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "class_metrics": {},
            "confusion_matrix": [],
        }

    if config.get("num_classes") is None:
        num_classes = max(
            int(torch.max(y_true).item()),
            int(torch.max(y_pred).item())
        ) + 1
    else:
        num_classes = config["num_classes"]

    # Fast confusion matrix construction
    indices = y_true * num_classes + y_pred
    cm = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).float()

    tp = torch.diag(cm)
    support = cm.sum(dim=1)      # true count per class
    predicted = cm.sum(dim=0)    # predicted count per class

    eps = 1e-12

    precision = tp / (predicted + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    weights = support / support.sum()

    # Masks for proper macro averaging
    precision_mask = predicted > 0
    recall_mask = support > 0
    f1_mask = support > 0

    precision_macro = precision[precision_mask].mean().item()
    recall_macro = recall[recall_mask].mean().item()
    f1_macro = f1[f1_mask].mean().item()

    metrics = {
        "accuracy": (tp.sum() / cm.sum()).item(),

        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,

        "precision_weighted": (precision * weights).sum().item(),
        "recall_weighted": (recall * weights).sum().item(),
        "f1_weighted": (f1 * weights).sum().item(),

        "class_metrics": {},
        "confusion_matrix": cm.long().tolist(),
    }

    for class_idx in range(num_classes):
        class_name = config["class_dict"].get(class_idx, f"class_{class_idx}")

        metrics["class_metrics"][class_name] = {
            "precision": precision[class_idx].item(),
            "recall": recall[class_idx].item(),
            "f1": f1[class_idx].item(),
            "support": int(support[class_idx].item()),
        }

    return metrics