"""Configurable loss functions for training.

Currently supports:
- CrossEntropyLoss  (with optional class weights).
- FocalLoss         (with optional class weights, gamma parameter and reduction method).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# LOSS FUNCTION FACTORY // Return a loss function based on a string identifier, with optional class weights and arguments
def get_loss_function(criterion_type="cross_entropy", weight=None, criterion_args={}, device=None):
    """Return a loss function based on the criterion_type string.

    Args:
        criterion_type: One of 'cross_entropy', 'focal'.
        weight: Optional class weights tensor.
        criterion_args: Dictionary of additional arguments for the loss function.
        device: Device to move weight tensor to.

    Returns:
        A loss function (nn.Module).
    """
    criterion_type = criterion_type.lower()
    
    if weight is not None and device is not None:
        weight = weight.to(device)

    if criterion_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    elif criterion_type == "focal":
        gamma = criterion_args.get("gamma", 2.0)
        reduction = criterion_args.get("reduction", "mean")
        return FocalLoss(gamma=gamma, weight=weight, reduction=reduction)
    else:
        raise ValueError(
            f"Unknown criterion_type: '{criterion_type}'. "
            "Choose from: 'cross_entropy', 'focal'."
        )


# FOCAL LOSS IMPLEMENTATION // for handling class imbalance by focusing on hard examples
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reduces the loss contribution from easy examples and focuses
    on hard misclassified examples.

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
        weight: Class weights tensor. Same as CrossEntropyLoss weight.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        """Initialise FocalLoss with focusing parameter and optional class weights."""
        super().__init__()
        # Validate inputs
        if not isinstance(gamma, (int, float)) or gamma < 0:
            raise ValueError(f"Invalid gamma value: {gamma}. Must be a non-negative number.")
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Invalid reduction value: '{reduction}'. Must be 'mean' or 'sum'.")
        
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """Compute focal loss given logits and targets."""
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        
        elif self.reduction == "sum":
            return focal_loss.sum()
