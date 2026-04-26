"""Tests for the BiLSTM feature extractor."""

import torch

from modules.models import BiLSTMFeatureExtractor


def test_bilstm_feature_extractor_output_shape():
    """Test that BiLSTMFeatureExtractor returns the expected feature shape."""
    model = BiLSTMFeatureExtractor(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=64,
    )

    D = torch.randint(1, 1000, (4, 10))
    DL = torch.tensor([10, 8, 6, 4])

    features = model(D, DL)

    assert features.shape == (4, 128)