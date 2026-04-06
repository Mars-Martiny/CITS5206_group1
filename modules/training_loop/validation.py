import torch
import torch.nn as nn
import torch.optim as optim

from .utility import _unpack_batch
from .metrics import _compute_classification_metrics


# VALIDATION FUNCTION // Evaluation funciton used for validation
def validate(config):
    """
    Validation function for evaluating the model on the validation set.
    Args:        
        - config (dict): A configuration dictionary containing the following keys:
            - model: The model to be evaluated.
            - valid_dl: The validation dataloader.
            - criterion: The loss function to be used for evaluation.
            - device: The device to run the evaluation on (e.g., 'cpu' or 'cuda').
            - num_classes: The number of classes in the classification task.
            - compute_val_metrics: Whether to compute detailed metrics (e.g., precision, recall, F1) in addition to loss and accuracy.
    Returns:
        - metrics (dict): A dictionary containing the calculated validation metrics, including:
            - accuracy, precision, recall, f1 macro, precision weighted, recall weighted, f1 weighted, loss
    """

    model = config["model"]
    criterion = config["criterion"]

    model.eval()

    total_loss, total_examples = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in config["valid_dl"]:
            logits, targets = _unpack_batch(batch, config)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = total_loss / max(total_examples, 1)
    all_preds = torch.cat(all_preds) if all_preds else torch.tensor([])
    all_targets = torch.cat(all_targets) if all_targets else torch.tensor([])

    metrics = _compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        num_classes=config["num_classes"],
    )
    metrics["loss"] = avg_loss

    return metrics