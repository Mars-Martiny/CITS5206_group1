"""BiGRU classifier model for sequence-based text classification."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiGRUClassifier(nn.Module):
    """BiGRU classifier module with architecture and forward pass only.

    Usage:
        logits = model(D, DL)

    input:
        D  : (batch_size, seq_len)
        DL : (batch_size,) 可选

    output:
        logits: (batch_size, num_classes)
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        """Initialize the model."""
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=num_classes,
        )

    def forward(self, D: torch.Tensor, DL: torch.Tensor | None = None) -> torch.Tensor:
        """Feed forward."""
        embedded = self.embedding(D)

        if DL is not None:
            packed = pack_padded_sequence(
                embedded,
                DL.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(embedded)

        # hidden shape:
        # (num_layers * 2, batch_size, hidden_dim)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        final_hidden = self.dropout(final_hidden)

        logits = self.classifier(final_hidden)

        return logits