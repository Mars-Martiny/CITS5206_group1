"""BiLSTM feature extractor for sequence-based text representations."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMFeatureExtractor(nn.Module):
    """Extract feature vectors from token sequences using a BiLSTM."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        """Initialize the BiLSTM feature extractor."""
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

    def forward(self, D: torch.Tensor, DL: torch.Tensor | None = None) -> torch.Tensor:
        """Extract BiLSTM feature vectors."""
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

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        features = torch.cat((forward_hidden, backward_hidden), dim=1)
        features = self.dropout(features)

        return features