import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from pipeline.run_feature_inference import run_feature_pipeline


class DummyFeatureExtractor(nn.Module):
    def forward(self, D, DL):
        return D.float()


class DummyClassifier:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        probs[:, 0] = 1.0
        return probs


def make_dataloader():
    D = torch.randint(1, 10, (6, 5))
    DL = torch.tensor([5, 5, 4, 3, 2, 1])
    Energy = torch.tensor([0, 1, 0, 1, 0, 1])
    Risk = torch.tensor([1, 1, 0, 0, 1, 0])

    dataset = TensorDataset(D, DL, Energy, Risk)
    return DataLoader(dataset, batch_size=2)


def test_feature_pipeline():
    train_dl = make_dataloader()
    test_dl = make_dataloader()

    result = run_feature_pipeline(
        feature_extractor=DummyFeatureExtractor(),
        classifier=DummyClassifier(),
        train_dl=train_dl,
        test_dl=test_dl,
        target="energy",
    )

    assert "predictions" in result
    assert "targets" in result
    assert "probabilities" in result