"""BiLSTM feature extractor for sequence-based text representations."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMFeatureExtractor(nn.Module):
    """Extract feature vectors from token sequences using a BiLSTM."""
    def __init__(self, bilstm_model):
        """Initialize the BiLSTM feature extractor."""
        super().__init__()
        self.embedding = bilstm_model.embedding
        self.lstm = bilstm_model.lstm
        self.dropout = bilstm_model.dropout

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