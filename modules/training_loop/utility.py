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
"""

import torch

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
# def _unpack_batch(batch, config):
#     """Unpack a dataloader batch and move tensors to the configured device.

#     Assumes batch structure: (D, DL, Energy, Risk) with length, or
#     (D, _, Energy, Risk) without length.
#     """
#     device = config["device"]
#     need_length = config["need_length"]
#     energy_model = config["energy_model"]

#     if need_length:
#         D, DL, Energy, Risk = batch
#         D = D.to(device)
#         DL = DL.to(device)
#         Energy = Energy.to(device)
#         Risk = Risk.to(device)
#         logits = config["model"](D, DL)
#     else:
#         D, _, Energy, Risk = batch
#         D = D.to(device)
#         Energy = Energy.to(device)
#         Risk = Risk.to(device)
#         logits = config["model"](D)

#     targets = Energy if energy_model else Risk
#     return logits, targets

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
