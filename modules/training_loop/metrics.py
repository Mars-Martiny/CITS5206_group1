import torch
import torch.nn as nn
import torch.optim as optim


# CLASSIFICATION METRICS COMPUTATION // Compute common classification metrics (accuracy, precision, recall, F1) from true and predicted labels
# Pure PyTorch implementation to avoid adding a hard dependency on sklearn.
def _compute_classification_metrics(y_true, y_pred, num_classes=None):
    """
    Args:
        - y_true: Tensor of shape (num_samples,) containing true class labels.
        - y_pred: Tensor of shape (num_samples,) containing predicted class labels.
        - num_classes: The number of classes in the classification task. 
            -> If None, it will be inferred from the data.
    Returns:
        - metrics (dict): Dictionary containing the computed metrics.
            -> currently includes: 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted'.
    
    Notes:
        - This function assumes that class labels are integer-encoded from 0 to num_classes-1.
        - If y_true is empty, all metrics will be returned as 0.0 to avoid division by zero errors.
    """
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
        }

    if num_classes is None:
        num_classes = int(torch.max(torch.cat([y_true, y_pred])).item()) + 1

    cm = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1.0

    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)

    eps = 1e-12
    precision_per_class = tp / (predicted + eps)
    recall_per_class = tp / (support + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    weights = support / (support.sum() + eps)

    accuracy = (tp.sum() / (cm.sum() + eps)).item()

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_per_class.mean().item(),
        "recall_macro": recall_per_class.mean().item(),
        "f1_macro": f1_per_class.mean().item(),
        "precision_weighted": (precision_per_class * weights).sum().item(),
        "recall_weighted": (recall_per_class * weights).sum().item(),
        "f1_weighted": (f1_per_class * weights).sum().item(),
    }

    return metrics

