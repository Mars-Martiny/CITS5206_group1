"""Utilities for handling class imbalance in training."""

import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels, num_classes=None):
    """Compute inverse-frequency class weights from a list of labels.

    Args:
        labels: List or tensor of integer class labels.
        num_classes: Total number of classes. Inferred from labels if None.

    Returns:
        Tensor of class weights of shape (num_classes,).
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


def make_weighted_sampler(labels, num_classes=None):
    """Create a WeightedRandomSampler for handling class imbalance.

    Args:
        labels: List or tensor of integer class labels for the training set.
        num_classes: Total number of classes. Inferred from labels if None.

    Returns:
        WeightedRandomSampler instance.
    """
    class_weights = compute_class_weights(labels, num_classes=num_classes)
    labels = torch.as_tensor(labels, dtype=torch.long)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )