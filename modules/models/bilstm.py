"""BiLSTM classifier model for sequence-based text classification."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    """Forward-only BiLSTM classifier.

    Expected usage in training loop:
        logits = model(D, DL)

    D:
        Tensor of token ids with shape (batch_size, seq_len)

    DL:
        Tensor of real sequence lengths with shape (batch_size,)

    Output:
        Raw logits with shape (batch_size, num_classes)
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

        self.lstm = nn.LSTM(
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
        """Forward pass.

        Parameters
        ----------
        D:
            Token id tensor, shape (batch_size, seq_len).

        DL:
            Sequence length tensor, shape (batch_size,).
            If provided, packed sequences are used so padding tokens are ignored.

        Returns:
        -------
        torch.Tensor
            Raw logits, shape (batch_size, num_classes).
        """
        embedded = self.embedding(D)

        if DL is not None:
            packed = pack_padded_sequence(
                embedded,
                DL.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # hidden shape:
        # (num_layers * num_directions, batch_size, hidden_dim)
        #
        # For BiLSTM:
        # hidden[-2] = final forward hidden state
        # hidden[-1] = final backward hidden state
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        final_hidden = self.dropout(final_hidden)

        logits = self.classifier(final_hidden)

        return logits