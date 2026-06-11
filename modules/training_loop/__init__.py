"""Training loop module exposing the top-level training entry point."""

from .config import _build_train_config
from .train_loop import train_model_loop

__all__ = ["training", "_build_train_config", "train_model_loop"]

"""
TRAINING LOOP MODULE // Files related to the main training loop and its components
- config: 
        Function to build the training configuration dictionary, which includes all necessary parameters and objects for training
- evaluate:
        Function that evaluates the model on the test set and returns the test loss and metrics, including checks against client performance requirements
- one_epoch: 
        Function that trains the model for one epoch and returns the training loss and accuracy
- validation: 
        Function that evaluates the model on the validation set and returns the validation loss and accuracy
- metrics: 
        Function for computing classification metrics (e.g., accuracy, precision, recall, F1) based on the true labels and predicted labels
- run_saving: 
        Functions for initializing the training history, appending metrics to the history, and saving the best model and run summary to disk at the end of training
- utility: 
        Utility functions for... 
                - safely getting the class name of an object 
                - converting various types of values (including tensors) to a format that can be easily saved in JSON or similar formats
                - unpacking batches from the dataloader and preparing them for model input
                - getting the learning rates from the optimizer's parameter groups
                - comparing the current metric value to the best metric value based on the specified mode (min or max)
                - normalising a class dictionary to ensure keys are integers and values are strings
"""


# EXPOSED TRAINING FUNCTION // main function to call for training a model, which builds the config and calls the main training loop
def training(
    model,
    energy_model=False,
    model_type="Simple",
    need_length=False,
    #
    optimiser=None,
    optimiser_args={},
    #
    scheduler=None,  # need to be defined outside
    scheduler_step_per_batch=False,
    #
    criterion_type="cross_entropy",
    criterion_weights=None,
    criterion_args={},
    #
    train_dl=None,
    valid_dl=None,
    test_dl=None,
    use_weighted_sampler=False,
    train_labels=None,
    #
    epochs=10,
    patience=3,
    num_classes=None,
    class_dict={},
    clip_grad_max_norm=1.0,
    #
    best_metric="loss",  # must be in: "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"
    best_metric_mode=None,
    #
    threshold=0.80,
    temperature=1.5,
    use_temperature=True,
    #
    parameters={},
    device="cpu",
    #
    compute_train_metrics=True,
    save=True,
    parent_dir="trained_models",
    run_name=None,
    #
    extra_config=None,
    requirements={},
    #
    log_leaderboard=True,
    leaderboard_dir="leaderboard",
    verbose=True,
    **_,
):
    """Build training config and run the full training loop.

    Args:
        model:          PyTorch model to train.
        energy_model:   True = predict energy type; False = predict risk type.
        model_type:     Label used for saving and logging.
        need_length:    Whether the model expects sequence lengths as input.

        optimiser:      Optimiser configuration.
                        Acceptable values:
                                - None (uses default Adam optimiser)
                                - str optimiser name
                                - configuration dictionary
                                - torch.optim.Optimizer object
        optimiser_args: Dictionary of additional arguments for the optimizer.

        scheduler:                  Learning rate scheduler. Defaults to StepLR.
        scheduler_step_per_batch:   Step scheduler per batch instead of per epoch.

        criterion_type:     Loss function type. One of 'cross_entropy', 'focal'. Defaults to 'cross_entropy'.
        criterion_weights:  Optional class weights for the loss function.
        criterion_args:     Dictionary of additional arguments for the loss function.

        train_dl:  DataLoader for training data.
        valid_dl:  DataLoader for validation data.
        test_dl:   DataLoader for test data.
        use_weighted_sampler:   Bool -> If True, use WeightedRandomSampler to handle class imbalance. Defaults to False.
        train_labels:           List or tensor of training labels. Required if use_weighted_sampler is True.

        epochs:              Number of training epochs.
        patience:            Early stopping patience in epochs.
        num_classes:         Number of output classes.
        class_dict:          Dictionary mapping class indices to class names of shape {int : str} // i.e. class index -> class name
        clip_grad_max_norm:  Max norm for gradient clipping.

        best_metric:         Metric used to select the best model checkpoint.
        best_metric_mode:    Towards 'min' or 'max' // Inferred from best_metric if None.

        threshold:        Confidence threshold for auto-classification.
        temperature:      Temperature for scaling logits...if use_temperature is True.
        use_temperature:  Whether to apply temperature scaling to logits.

        parameters:  The training parameters in a dictionary format.
        device:      Device string, e.g. 'cpu' or 'cuda'.

        compute_train_metrics:  Whether to compute metrics on the training set. Defaults to True.
        save:                   Whether to save model artifacts after training.
        parent_dir:             Directory to save model artifacts.
        run_name:               Optional name for the run.
            
        extra_config:  Optional dict of additional config keys to merge.
        requirements:  Optional client performance requirements dict, defaults to {}. 
            Pass None to disable check. Keys:
                - confidence_threshold: {"high": float, "medium": float} (values >1 treated as %)
                - high_threshold: min fraction of predictions in high-confidence tier (default 0.70)
                - fatal_accuracy: min recall on true fatal-class samples (default 0.95)
                - f1_target: {class_index: min_f1} — use 0.0 to mark a class as having no target
        
        log_leaderboard:    Whether to append this run to the leaderboard CSV. Defaults to True. // If True, requires save to be True for logging.
        leaderboard_dir:    Directory for leaderboard.csv and owner.conf. Defaults to 'leaderboard'.
        verbose:            Enable printing the training loop message. Defaults to True.

    Returns:
        Run summary dictionary with history, best epoch, and best metric value.
    """
    # All top-level inputs are collected into `config` and that config is passed everywhere else.
    # This keeps the function signatures clean and makes it easy to add new parameters without needing to change a lot of function signatures.
    train_config = _build_train_config(
        model=model,
        energy_model=energy_model,
        model_type=model_type,
        need_length=need_length,
        #
        optimiser=optimiser,
        optimiser_args=optimiser_args,
        #
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
        #
        criterion_type=criterion_type,
        criterion_weights=criterion_weights,
        criterion_args=criterion_args,
        #
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        use_weighted_sampler=use_weighted_sampler,
        train_labels=train_labels,
        #
        epochs=epochs,
        patience=patience,
        num_classes=num_classes,
        class_dict=class_dict,
        clip_grad_max_norm=clip_grad_max_norm,
        #
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        #
        threshold=threshold,
        temperature=temperature,
        use_temperature=use_temperature,
        #
        parameters=parameters,
        device=device,
        #
        compute_train_metrics=compute_train_metrics,
        save=save,
        parent_dir=parent_dir,
        run_name=run_name,
        #
        extra_config=extra_config,
        requirements=requirements,
    )

    train_config["log_leaderboard"] = log_leaderboard
    train_config["leaderboard_dir"] = leaderboard_dir
    train_config["verbose"] = verbose

    return train_model_loop(train_config)