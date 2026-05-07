"""BiLSTM training entry point for the shared pipeline."""

from __future__ import annotations

import ast
from typing import Any

import torch
import torch.optim as optim

from modules.data_loader import df_to_dataloader
from modules.encoding import LabelEncoder, VocabEncoder
from modules.models import BiLSTMClassifier
from modules.training_loop import training


def _normalise_tokens(value: Any) -> list[str]:
    """Return one row of token data as a list of strings.

    Token columns in notebooks may already contain lists. When loaded back from
    CSV, the same values can appear as string representations of lists, so this
    helper keeps the BiLSTM entry point usable in both cases.
    """
    if isinstance(value, list):
        return [str(token) for token in value]

    if isinstance(value, tuple):
        return [str(token) for token in value]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped.split()
        if isinstance(parsed, (list, tuple)):
            return [str(token) for token in parsed]
        return stripped.split()

    if value is None:
        return []

    try:
        if value != value:  # NaN check without requiring pandas at runtime.
            return []
    except TypeError:
        pass

    return [str(value)]


def prepare_bilstm_dataframe(
    df,
    *,
    tokens_col: str,
    label_col: str,
    risk_col: str | None = None,
    vocab_encoder: VocabEncoder | None = None,
    label_encoder: LabelEncoder | None = None,
    risk_encoder: LabelEncoder | None = None,
    max_length: int = 128,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    encoded_tokens_col: str = "bilstm_token_ids",
    encoded_label_col: str = "bilstm_energy_label",
    encoded_risk_col: str = "bilstm_risk_label",
):
    """Encode a dataframe for BiLSTM training.

    Args:
        df: DataFrame containing tokenized text and labels.
        tokens_col: Column containing token lists or token-list strings.
        label_col: Target label column for energy classification.
        risk_col: Optional risk target column. If omitted, a dummy risk column
            is created so the shared ``(D, DL, Energy, Risk)`` batch format is
            still satisfied.
        vocab_encoder: Optional fitted vocabulary encoder. If omitted, a new
            one is fitted on ``df``.
        label_encoder: Optional fitted energy label encoder. If omitted, a new
            one is fitted on ``df``.
        risk_encoder: Optional fitted risk label encoder. If omitted and
            ``risk_col`` is provided, a new one is fitted on ``df``.
        max_length: Maximum encoded sequence length.
        min_freq: Minimum token frequency for new vocabulary fitting.
        max_vocab_size: Optional maximum vocabulary size.
        encoded_tokens_col: Output token-id column name.
        encoded_label_col: Output energy-label column name.
        encoded_risk_col: Output risk-label column name.

    Returns:
        Tuple of ``(encoded_df, vocab_encoder, label_encoder, risk_encoder)``.
    """
    encoded_df = df.copy()
    token_sequences = encoded_df[tokens_col].apply(_normalise_tokens).tolist()

    if vocab_encoder is None:
        vocab_encoder = VocabEncoder(
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )
        vocab_encoder.fit(token_sequences)

    encoded_sequences = []
    for token_sequence in token_sequences:
        token_ids = vocab_encoder.encode_tokens(token_sequence)[:max_length]
        if not token_ids:
            token_ids = [vocab_encoder.unk_id]
        encoded_sequences.append(token_ids)

    encoded_df[encoded_tokens_col] = encoded_sequences

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(encoded_df[label_col].tolist())
    encoded_df[encoded_label_col] = label_encoder.encode_many(
        encoded_df[label_col].tolist(),
    )

    if risk_col is None:
        encoded_df[encoded_risk_col] = 0
    else:
        if risk_encoder is None:
            risk_encoder = LabelEncoder()
            risk_encoder.fit(encoded_df[risk_col].tolist())
        encoded_df[encoded_risk_col] = risk_encoder.encode_many(
            encoded_df[risk_col].tolist(),
        )

    return encoded_df, vocab_encoder, label_encoder, risk_encoder


def build_bilstm_dataloader(
    df,
    *,
    tokens_col: str = "bilstm_token_ids",
    energy_col: str = "bilstm_energy_label",
    risk_col: str = "bilstm_risk_label",
    batch_size: int = 32,
    pad_id: int = 0,
    shuffle: bool = True,
):
    """Build a shared-pipeline DataLoader for the BiLSTM model.

    The returned batches have the exact ``(D, DL, Energy, Risk)`` structure
    expected by ``modules.training_loop.utility._unpack_batch``.
    """
    return df_to_dataloader(
        df,
        tokens_col=tokens_col,
        energy_col=energy_col,
        risk_col=risk_col,
        batch_size=batch_size,
        pad_id=pad_id,
        shuffle=shuffle,
    )


def prepare_bilstm_dataloaders(
    train_df,
    valid_df,
    test_df,
    *,
    tokens_col: str,
    label_col: str,
    risk_col: str | None = None,
    batch_size: int = 32,
    max_length: int = 128,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
):
    """Prepare train, validation, and test DataLoaders for BiLSTM."""
    train_encoded, vocab_encoder, label_encoder, risk_encoder = (
        prepare_bilstm_dataframe(
            train_df,
            tokens_col=tokens_col,
            label_col=label_col,
            risk_col=risk_col,
            max_length=max_length,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )
    )

    valid_encoded, _, _, _ = prepare_bilstm_dataframe(
        valid_df,
        tokens_col=tokens_col,
        label_col=label_col,
        risk_col=risk_col,
        vocab_encoder=vocab_encoder,
        label_encoder=label_encoder,
        risk_encoder=risk_encoder,
        max_length=max_length,
    )
    test_encoded, _, _, _ = prepare_bilstm_dataframe(
        test_df,
        tokens_col=tokens_col,
        label_col=label_col,
        risk_col=risk_col,
        vocab_encoder=vocab_encoder,
        label_encoder=label_encoder,
        risk_encoder=risk_encoder,
        max_length=max_length,
    )

    train_dl = build_bilstm_dataloader(
        train_encoded,
        batch_size=batch_size,
        pad_id=vocab_encoder.pad_id,
        shuffle=True,
    )
    valid_dl = build_bilstm_dataloader(
        valid_encoded,
        batch_size=batch_size,
        pad_id=vocab_encoder.pad_id,
        shuffle=False,
    )
    test_dl = build_bilstm_dataloader(
        test_encoded,
        batch_size=batch_size,
        pad_id=vocab_encoder.pad_id,
        shuffle=False,
    )

    metadata = {
        "vocab_encoder": vocab_encoder,
        "label_encoder": label_encoder,
        "risk_encoder": risk_encoder,
        "train_df": train_encoded,
        "valid_df": valid_encoded,
        "test_df": test_encoded,
    }
    return train_dl, valid_dl, test_dl, metadata


def run_bilstm_training(
    train_df,
    valid_df,
    test_df,
    *,
    tokens_col: str,
    label_col: str,
    risk_col: str | None = None,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.5,
    batch_size: int = 32,
    max_length: int = 128,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    epochs: int = 50,
    patience: int = 3,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    **training_kwargs,
):
    """Instantiate BiLSTMClassifier and run it through the main training loop."""
    train_dl, valid_dl, test_dl, metadata = prepare_bilstm_dataloaders(
        train_df,
        valid_df,
        test_df,
        tokens_col=tokens_col,
        label_col=label_col,
        risk_col=risk_col,
        batch_size=batch_size,
        max_length=max_length,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )

    vocab_encoder = metadata["vocab_encoder"]
    label_encoder = metadata["label_encoder"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = BiLSTMClassifier(
        vocab_size=vocab_encoder.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=label_encoder.num_classes,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=vocab_encoder.pad_id,
    ).to(device)

    optimiser = training_kwargs.pop(
        "optimiser",
        optim.Adam(model.parameters(), lr=lr),
    )

    run_summary = training(
        model=model,
        energy_model=True,
        model_type="BiLSTM",
        need_length=True,
        optimiser=optimiser,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        epochs=epochs,
        patience=patience,
        best_metric=training_kwargs.pop("best_metric", "f1_macro"),
        num_classes=label_encoder.num_classes,
        class_dict=label_encoder.id_to_label,
        device=device,
        parameters={
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "max_length": max_length,
            "min_freq": min_freq,
            "max_vocab_size": max_vocab_size,
            "lr": lr,
        },
        extra_config={"bilstm_metadata": metadata},
        **training_kwargs,
    )
    return run_summary