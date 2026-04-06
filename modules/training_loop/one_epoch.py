import torch
import torch.nn as nn
import torch.optim as optim

from .utility import _unpack_batch, _get_learning_rates
from .metrics import _compute_classification_metrics



# SINGLE EPOCH TRAINING LOOP // returning training accuracy and loss of one training epoch
def train_one_epoch(config):
    """
    Train the model for one epoch.
    Args:
        - config: A dictionary containing the following keys:
            - model: The PyTorch model to train.
            - optimiser: The optimizer to use for training.
            - scheduler: The learning rate scheduler to use (optional).
            - criterion: The loss function to use for training.
            - train_dl: The DataLoader for the training data.
            - num_classes: The number of classes in the classification task.
            - clip_grad_max_norm: The maximum norm for gradient clipping (optional).
    Returns:
        - metrics (dict): A dictionary containing the calculated validation metrics, listed below, and the current learning rate:
            - accuracy, precision, recall, f1 macro, precision weighted, recall weighted, f1 weighted, loss
    """

    model = config["model"]
    optimiser = config["optimiser"]
    scheduler = config["scheduler"]
    criterion = config["criterion"]

    model.train()

    total_loss, total_examples = 0.0, 0
    all_preds, all_targets = [], []

    for batch in config["train_dl"]:
        optimiser.zero_grad(set_to_none=True)

        logits, targets = _unpack_batch(batch, config)
        loss = criterion(logits, targets)
        loss.backward()

        if config["clip_grad_max_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_max_norm"])

        optimiser.step()

        if scheduler is not None and config["scheduler_step_per_batch"]:
            scheduler.step()

        preds = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    avg_loss = total_loss / max(total_examples, 1)
    all_preds = torch.cat(all_preds) if all_preds else torch.tensor([])
    all_targets = torch.cat(all_targets) if all_targets else torch.tensor([])

    # Compute classification metrics
    metrics = _compute_classification_metrics(y_true=all_targets, y_pred=all_preds, 
                                              num_classes=config["num_classes"])
    metrics["loss"] = avg_loss
    metrics["lr"] = _get_learning_rates(config)

    return metrics