"""Tests for the BiLSTM feature extractor."""

import torch

from modules.models import BiLSTMFeatureExtractor, BiLSTMClassifier


def test_bilstm_feature_extractor_output_shape():
    """Test that BiLSTMFeatureExtractor returns the expected feature shape."""

    bilstm_model = BiLSTMClassifier(
        vocab_size=20,
        embedding_dim=8,
        hidden_dim=16,
        num_classes=2,
    )

    extractor = BiLSTMFeatureExtractor(bilstm_model)

    D = torch.randint(1, 20, (4, 10))
    DL = torch.tensor([10, 8, 6, 4])

    features = extractor(D, DL)

    assert features.shape == (4, 32)