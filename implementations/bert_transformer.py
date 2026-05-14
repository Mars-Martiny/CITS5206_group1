"""Run BERT-style transformer experiments for Energy Type and Risk classification."""

from __future__ import annotations

import torch
import torch.optim as optim

from modules.data_loader.bert_loader import df_to_bert_dataloader
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_tokenizer import BertTokenizerWrapper
from modules.embedding.bert_embedding import BertEmbeddingBackend
from modules.encoding.label_encoder import LabelEncoder
from modules.models.bert_classifier import BertClassifier
from modules.training_loop import training

BEST_SAFETYBERT_CONFIG = {
    "fine_tune": True,
    "pooling": "cls",
    "learning_rate": 2e-5,
    "dropout": 0.275,
    "weight_decay": 0.01,
    "max_length": 160,
    "batch_size": 8,
    "epochs": 6,
    "use_class_weights": False,
    "scheduler_config": {
        "name": "ReduceLROnPlateau",
        "monitor": "f1_macro",
        "mode": "max",
        "factor": 0.5,
        "patience": 1,
        "min_lr": 1e-7,
        "step_per_batch": False,
        },
    }

def get_best_available_device() -> torch.device:
    """Return the best available PyTorch device.

    Priority:
    1. CUDA
    2. Apple Silicon MPS
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def encode_label_column(train_df, valid_df, test_df, label_col):
    """Fit label encoder on train labels and transform train/valid/test.
    
    Args:
        train_df: Training dataframe.
        valid_df: Validation dataframe.
        test_df: Test dataframe.
        label_col: Name of the column containing label data.

    Returns:
        Tuple of (train_df, valid_df, test_df, label_encoder, class_names) where the dataframes have the label column encoded as integers, label_encoder is the fitted LabelEncoder instance, and class_names is a list mapping label IDs to original label names.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[label_col].astype(str).tolist())

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df[label_col] = label_encoder.encode_many(
        train_df[label_col].astype(str).tolist()
    )
    valid_df[label_col] = label_encoder.encode_many(
        valid_df[label_col].astype(str).tolist()
    )
    test_df[label_col] = label_encoder.encode_many(
        test_df[label_col].astype(str).tolist()
    )

    class_names = [
        label_encoder.id_to_label[i]
        for i in range(label_encoder.num_classes)
    ]

    return train_df, valid_df, test_df, label_encoder, class_names


def build_class_weights(train_df, label_col: str, device: torch.device):
    """Build inverse-frequency class weights on the target device.
    
    Args:
        train_df: Training dataframe.
        label_col: Name of the column containing label data.
        device: PyTorch device to place the weights on.
    
    Returns:
        Tensor of class weights.
    """
    class_counts = train_df[label_col].value_counts().sort_index()
    weights = 1.0 / class_counts
    weights = weights / weights.mean()

    return torch.tensor(
        weights.values,
        dtype=torch.float,
        device=device,
    )


def run_bert_experiment(
    train_df,
    valid_df,
    test_df,
    text_col,
    label_col,
    run_name,
    fine_tune=False,
    pooling="mean",
    batch_size=8,
    epochs=5,
    learning_rate=None,
    dropout=0.2,
    embedding_dropout=0.1,
    max_length=160,
    use_class_weights=False,
    weight_decay=0.01,
    threshold=0.8,
    patience=2,
    model_name="bert-base-uncased",
    tokenizer_name=None,
    model_type="BERT",
    target_type="energy",
    scheduler_config=None,
):
    """Train and evaluate a BERT-style classifier for one target label.

    The `dropout` argument refers to classifier-head dropout. The `embedding_dropout` argument is applied to transformer token embeddings
    before CLS/mean pooling.

    For SafetyBERT, use:
        model_name="adanish91/safetybert"
        tokenizer_name="bert-base-uncased"
        model_type="SafetyBERT"
        
    Arguments:
        train_df: Training dataframe.
        valid_df: Validation dataframe.
        test_df: Test dataframe.
        text_col: Name of the column containing text data.
        label_col: Name of the column containing label data.
        run_name: Name for this training run.
        fine_tune: Whether to fine-tune BERT or keep it frozen.
        pooling: Pooling strategy for sentence embedding, either "cls" or "mean".
        batch_size: Batch size for dataloaders.
        epochs: Maximum number of training epochs.
        learning_rate: Learning rate. If None, uses 2e-5 when fine-tuning and 1e-4 when frozen.
        dropout: Dropout rate for classifier head.
        embedding_dropout: Dropout rate applied to transformer token embeddings before pooling. This is separate from classifier-head dropout.
        max_length: Maximum sequence length for BERT tokenizer.
        use_class_weights: Whether to use class weights in the loss function.
        weight_decay: Weight decay factor for AdamW.
        threshold: Confidence threshold for auto-classification analysis.
        patience: Number of epochs with no improvement on the validation metric before early stopping.
        model_name: Hugging Face model identifier for the BERT variant to use.
        tokenizer_name: Hugging Face model identifier for the tokenizer to use. If None, defaults to the same as model_name.
        model_type: String identifier for the model type, used in metadata and save names.
        target_type: String identifier for the target type ("energy" or "risk"), used in metadata and save names.
        scheduler_config: Optional dictionary specifying the learning rate scheduler configuration. If None, no scheduler is used. Expected keys include "name" for the scheduler type and other scheduler-specific parameters.
    
    Returns:
        Run summary returned by the shared training pipeline.
    """
    if target_type not in {"energy", "risk"}:
        raise ValueError("target_type must be either 'energy' or 'risk'.")

    device = get_best_available_device()

    train_df, valid_df, test_df, label_encoder, class_names = encode_label_column(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_col=label_col,
    )

    bert_config = BertEmbeddingConfig(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dropout=embedding_dropout,
        pooling=pooling,
        fine_tune=fine_tune,
    )

    tokenizer_wrapper = BertTokenizerWrapper(bert_config)

    train_dl = df_to_bert_dataloader(
        train_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_dl = df_to_bert_dataloader(
        valid_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dl = df_to_bert_dataloader(
        test_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=False,
    )

    embedding_backend = BertEmbeddingBackend(bert_config)

    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=label_encoder.num_classes,
        dropout=dropout,
    ).to(device)

    if learning_rate is None:
        learning_rate = 2e-5 if fine_tune else 1e-4

    optimiser = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion_weights = None
    if use_class_weights:
        criterion_weights = build_class_weights(
            train_df=train_df,
            label_col=label_col,
            device=device,
        )

    scheduler_step_per_batch = (
        scheduler_config.get("step_per_batch", False)
        if scheduler_config
        else False
    )

    run_summary = training(
        model=model,
        energy_model=(target_type == "energy"),
        model_type=model_type,
        need_length=False,

        optimiser=optimiser,
        optimiser_args=None,

        scheduler=scheduler_config,
        scheduler_step_per_batch=scheduler_step_per_batch,

        criterion_type="cross_entropy",
        criterion_weights=criterion_weights,
        criterion_args=None,

        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,

        epochs=epochs,
        patience=patience,
        num_classes=label_encoder.num_classes,
        class_dict=label_encoder.id_to_label,
        clip_grad_max_norm=1.0,

        best_metric="f1_macro",
        best_metric_mode=None,

        threshold=threshold,
        temperature=1.0,
        use_temperature=False,

        parameters={
            "model_type": model_type,
            "target_type": target_type,
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "pooling": pooling,
            "fine_tune": fine_tune,
            "batch_size": batch_size,
            "label_col": label_col,
            "text_col": text_col,
            "learning_rate": learning_rate,
            "classifier_dropout": dropout,
            "embedding_dropout": embedding_dropout,
            "max_length": max_length,
            "use_class_weights": use_class_weights,
            "weight_decay": weight_decay,
            "threshold": threshold,
            "patience": patience,
            "scheduler_config": scheduler_config,
            "device": str(device),
        },
        device=device,

        compute_train_metrics=False,
        save=True,
        parent_dir="trained_models",
        run_name=run_name,

        extra_config={
            "class_names": class_names,
            "label_col": label_col,
            "text_col": text_col,
            "target_type": target_type,
            "model_type": model_type,
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "pooling": pooling,
            "fine_tune": fine_tune,
        },
    )

    return run_summary


def run_safetybert_experiment(
    train_df,
    valid_df,
    test_df,
    text_col,
    label_col,
    run_name,
    **kwargs,
):
    """Run a SafetyBERT experiment using the shared BERT pipeline.

    SafetyBERT is loaded as the pretrained encoder, while the rest of the
    pipeline remains identical to the standard BERT experiment.
    """
    return run_bert_experiment(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        text_col=text_col,
        label_col=label_col,
        run_name=run_name,
        model_name="adanish91/safetybert",
        tokenizer_name="bert-base-uncased",
        model_type="SafetyBERT",
        **kwargs,
    )

def run_safetybert_best_experiment(
    train_df,
    valid_df,
    test_df,
    text_col,
    label_col,
    run_name,
    target_type="energy",
    **kwargs,
):
    """Run SafetyBERT using the best-found hyperparameter configuration."""
    params = {
        "model_name": "adanish91/safetybert",
        "tokenizer_name": "bert-base-uncased",
        "model_type": "SafetyBERT",
        "target_type": target_type,
        **BEST_SAFETYBERT_CONFIG,
    }

    params.update(kwargs)

    return run_bert_experiment(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        text_col=text_col,
        label_col=label_col,
        run_name=run_name,
        **params,
    )