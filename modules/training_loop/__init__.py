from .config import _build_train_config
from .train_loop import train_model_loop

'''
TRAINING LOOP MODULE // Files related to the main training loop and its components
- config: 
        Function to build the training configuration dictionary, which includes all necessary parameters and objects for training
- one_epoch: 
        Function that trains the model for one epoch and returns the training loss and accuracy
- validation: 
        Function that evaluates the model on the validation set and returns the validation loss and accuracy
- metrics: 
        Function for computing classification metrics (e.g., accuracy, precision, recall, F1) based on the true labels and predicted labels
- run_saving: 
        Functions for initializing the training history, appending metrics to the history, and saving the best model and run summary to disk at the end of training
- utility: 
        Utility functions for 
                - safely getting the class name of an object 
                - converting various types of values (including tensors) to a format that can be easily saved in JSON or similar formats
                - unpacking batches from the dataloader and preparing them for model input
                - getting the learning rates from the optimizer's parameter groups
                - comparing the current metric value to the best metric value based on the specified mode (min or max)


Functionality notes (): 
 - Support for different loss functions 
    -> currently supports any loss function that can be called as `criterion(preds, labels)` and returns a scalar loss value. 
        -> Could add support for more complex loss functions that require additional inputs (e.g., class weights, sample weights, etc.)
 - More detailed model saving 
    -> model saving is now more detailed
 - More detailed logging (e.g., using TensorBoard or a logging library instead of print statements)
    -> currently uses print statements for logging, but could be extended to use a more robust logging framework or TensorBoard for better visualization of training progress and metrics
 - Support for resuming training from a checkpoint
    -> currently does not support resuming training from a checkpoint, 
        -> could be extended to allow loading a saved model state dict and training history to continue training from where it left off
 - More flexible learning rate scheduling (e.g., support for different schedulers, or custom scheduling logic)
    -> currently supports a default StepLR scheduler, or use of a custom scheduler if provided. 
        -> could be extended to support a wider range of built-in schedulers, or allow for custom scheduling logic to be implemented by the user
 - ....etc.
'''


# EXPOSED TRAINING FUNCTION // main function to call for training a model, which builds the config and calls the main training loop
def training(
             model,
             optimiser=None,
             train_dl=None,
             valid_dl=None,
             epochs=10,
             device="cpu",
             patience=3,
             criterion_weights=None,
             model_type="Simple",
             save=True,
             scheduler=None,        # need to be defined outside
             criterion=None,        # need to be defined outside
             need_length=False,
             energy_model=False,
             best_metric="loss",    # must be in: "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"
             best_metric_mode=None,
             clip_grad_max_norm=1.0,
             scheduler_step_per_batch=False,
             save_dir="trained_models",
             run_name=None,
             compute_train_metrics=False,
             num_classes=None,
             extra_config=None,
            ):
    
    # All top-level inputs are collected into `config` and that config is passed everywhere else.
    # This keeps the function signatures clean and makes it easy to add new parameters without needing to change a lot of function signatures.
    train_config = _build_train_config(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        epochs=epochs,
        device=device,
        patience=patience,
        criterion_weights=criterion_weights,
        model_type=model_type,
        save=save,
        optimiser=optimiser,
        scheduler=scheduler,
        criterion=criterion,
        need_length=need_length,
        energy_model=energy_model,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        clip_grad_max_norm=clip_grad_max_norm,
        scheduler_step_per_batch=scheduler_step_per_batch,
        save_dir=save_dir,
        run_name=run_name,
        compute_train_metrics=compute_train_metrics,
        num_classes=num_classes,
        extra_config=extra_config,
    )

    return train_model_loop(train_config)