"""
UTILITY FUNCTIONS FOR TRAINING LOOP
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


# GET CLASS NAME // Safely get the class name of an object, returning None if the object is None
def _safe_class_name(obj):
    return obj.__class__.__name__ if obj is not None else None


# CONVERT METRICS TO SERIALISABLE FORMAT // Convert various types of values (including tensors) to a format that can be easily saved in JSON or similar formats
def _serialise_value(value):
    """
    Best-effort conversion for config/checkpoint metadata.
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
    """
    Assumes the same batch structure as your current code:
      - with length:  D, DL, Energy, Risk
      - without length: D, _, Energy, Risk
    """
    device = config["device"]
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
    """
    Args:
        - config: A dictionary containing the training configuration, which must include the 'optimiser'
    Returns:
        - list: A list of learning rates corresponding to each parameter group in the optimizer.
    """

    optimiser = config["optimiser"]
    
    return [group["lr"] for group in optimiser.param_groups]


# BEST MODEL COMPARISON // Compare the current metric value to the best metric value based on the specified mode (min or max)
def _is_better(current, best, mode):
    """
    Args:
        - current: The current metric value to compare.
        - best: The best metric value seen so far (can be None if no best value has been set).
        - mode: A string indicating whether a lower value is better ('min') or a higher value is better ('max').
    Returns:
        - bool: True if the current value is better than the best value according to the specified mode, False otherwise.
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