"""TF-IDF vectorisation and feed-forward classification for text inputs."""

import math
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader


class TFIDFVectorizer:
    """Fits on tokenized documents and transforms them to TF-IDF feature vectors.

    Attributes:
        vocab: Mapping from term string to column index in the feature matrix.
        df: Document frequency counter for each term in the corpus.
        N: Total number of documents seen during fitting.
    """

    def fit(self, tokenized_docs: list[list[str]]) -> "TFIDFVectorizer":
        """Builds vocabulary and document-frequency counts from a corpus.

        Terms that appear in fewer than two documents are excluded from the
        vocabulary to reduce noise and feature dimensionality.

        Args:
            tokenized_docs: List of documents, where each document is a list
                of string tokens.

        Returns:
            The fitted vectorizer instance (enables method chaining).
        """
        self.vocab = {}
        df = Counter()
        for doc in tokenized_docs:
            for term in set(doc):
                df[term] += 1
        self.vocab = {t: i for i, t in enumerate(t for t, c in df.items() if c >= 2)}
        self.df = df
        self.N = len(tokenized_docs)
        return self

    def transform(self, tokenized_docs: list[list[str]]) -> torch.Tensor:
        """Converts tokenized documents into a TF-IDF feature matrix.

        Uses smoothed IDF: idf(t) = log(N / (df(t) + 1)) + 1. Terms not
        present in the fitted vocabulary are silently ignored.

        Args:
            tokenized_docs: List of documents to transform, where each
                document is a list of string tokens.

        Returns:
            A float tensor of shape ``(num_docs, vocab_size)`` where each row
            is the TF-IDF feature vector for the corresponding document.
        """
        vectors = torch.zeros(len(tokenized_docs), len(self.vocab))
        for doc_id, doc in enumerate(tokenized_docs):
            counter = Counter(doc)
            total = len(doc)
            for term in np.unique(doc):
                if term not in self.vocab:
                    continue
                tf = counter[term] / total
                idf = math.log(self.N / (self.df[term] + 1)) + 1
                vectors[doc_id, self.vocab[term]] = tf * idf
        return vectors


class TFIDFClassifier(nn.Module):
    """Feed-forward classifier that operates on TF-IDF feature vectors.

    Architecture: Linear → ReLU → Dropout(0.3) → Linear.

    Args:
        vocab_size: Dimensionality of the input TF-IDF feature vectors.
        num_classes: Number of output classes.
        hidden_dim: Width of the hidden layer. Defaults to 256.
    """

    def __init__(self, vocab_size: int, num_classes: int, hidden_dim: int = 256):
        """Initialises the classifier network layers.

        Args:
            vocab_size: Dimensionality of the input TF-IDF feature vectors.
            num_classes: Number of output classes.
            hidden_dim: Width of the hidden layer. Defaults to 256.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits for a batch of TF-IDF feature vectors.

        Args:
            x: Input tensor of shape ``(batch_size, vocab_size)``.

        Returns:
            Logit tensor of shape ``(batch_size, num_classes)``.
        """
        return self.net(x.float())


def build_tfidf_dataloader(
    tfidf_vectors: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Wraps TF-IDF vectors and labels in a DataLoader.

    Produces batches in the ``(D, DL, Energy, Risk)`` format expected by the
    shared training loop. The ``DL`` and ``Energy`` slots are filled with
    zero-valued placeholder tensors.

    Args:
        tfidf_vectors: Float tensor of shape ``(num_samples, vocab_size)``
            containing pre-computed TF-IDF features.
        labels: Long tensor of shape ``(num_samples,)`` containing class
            indices.
        batch_size: Number of samples per batch. Defaults to 32.
        shuffle: Whether to shuffle the dataset each epoch. Defaults to True.

    Returns:
        A DataLoader yielding 4-tuples of
        ``(tfidf_vectors, dummy_dl, labels, dummy_energy)``.
    """
    dummy = torch.zeros(len(tfidf_vectors), dtype=torch.long)  # placeholder DL
    dataset = TensorDataset(tfidf_vectors, dummy, labels, dummy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)