"""Inference module for running model predictions on a dataset."""

import torch
import torch.nn.functional as F


def run_inference(config, dataloader=None):
    """Run model inference on a dataset.

    Collect predictions and probabilities for downstream metric computation.

    Args:
        config (dict): Configuration dictionary containing model, device,
            temperature settings, and optional test dataloader.
        dataloader: Optional dataloader override. If None, falls back to
            config["test_dl"]. If still None, only model.eval() is called.

    Returns:
        Dictionary with all_preds, all_targets, all_probs, and total_examples
        if a dataloader is available; otherwise None.
    """
    model = config["model"]
    device = config.get("device", "cpu")
    temperature = config.get("temperature", 1.0)
    use_temperature = config.get("use_temperature", False)

    if use_temperature and temperature <= 0:
        raise ValueError("temperature must be > 0 when use_temperature=True")

    from modules.training_loop.utility import _unpack_batch

    model.to(device)
    model.eval()

    dl = dataloader or config.get("test_dl")
    if dl is None:
        return None

    all_preds, all_targets, all_probs = [], [], []
    total_examples = 0

    with torch.no_grad():
        for batch in dl:
            logits, targets = _unpack_batch(batch, config)

            if use_temperature:
                logits = logits / temperature

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            batch_size = targets.size(0)
            total_examples += batch_size

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_probs.append(probs.detach().cpu())

    return {
        "all_preds": torch.cat(all_preds) if all_preds else torch.tensor([], dtype=torch.long),
        "all_targets": torch.cat(all_targets) if all_targets else torch.tensor([], dtype=torch.long),
        "all_probs": torch.cat(all_probs) if all_probs else torch.empty((0, config["num_classes"])),
        "total_examples": total_examples,
    }