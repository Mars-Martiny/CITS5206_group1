"""Tests for the BiLSTM classifier."""

import torch

from modules.models import BiLSTMClassifier


def test_bilstm_classifier_output_shape():
    """Test that BiLSTMClassifier returns logits with the expected shape."""
    model = BiLSTMClassifier(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=64,
        num_classes=5,
    )

    D = torch.randint(1, 1000, (4, 10))
    DL = torch.tensor([10, 8, 6, 4])

    logits = model(D, DL)

    assert logits.shape == (4, 5)


def test_bilstm_classifier_forward_without_lengths():
    """Test that BiLSTMClassifier can run without sequence lengths."""
    model = BiLSTMClassifier(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=64,
        num_classes=5,
    )

    D = torch.randint(1, 1000, (4, 10))

    logits = model(D)

    assert logits.shape == (4, 5)