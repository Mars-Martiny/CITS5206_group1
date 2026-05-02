"""Wrapper to integrate feature extractor + classical classifier into inference pipeline."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FeatureClassifierWrapper(nn.Module):
    """Wrap a feature extractor and classical classifier as a PyTorch model."""

    def __init__(self, feature_extractor: nn.Module, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, D, DL=None):
        """Return logits compatible with run_inference."""
        self.feature_extractor.eval()

        with torch.no_grad():
            if DL is not None:
                features = self.feature_extractor(D, DL)
            else:
                features = self.feature_extractor(D)

        X = features.detach().cpu().numpy()

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
        else:
            preds = self.classifier.predict(X)
            n_classes = len(set(preds))
            probs = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probs[i, p] = 1.0

        return torch.tensor(probs, dtype=torch.float32)