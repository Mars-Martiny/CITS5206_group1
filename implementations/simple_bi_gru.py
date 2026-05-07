"""Bidirectional GRU classifier for single-text incident classification."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader


class BiGRUClassifier(nn.Module):
    """Single-sequence BiGRU classifier compatible with the shared training loop.

    Encodes a padded token-index sequence with a bidirectional GRU and feeds the
    concatenated final hidden states into a linear classification head.

    Args:
        vocab_size: Vocabulary size including ``<pad>`` (index 0) and ``<unk>``.
        embedding_dim: Dimensionality of token embeddings.
        hidden_dim: Hidden size per GRU direction.
        num_classes: Number of output classes.
        emb_table: Optional numpy array of shape ``(vocab_size, embedding_dim)``
            used to initialise the embedding layer. If ``None``, embeddings are
            randomly initialised.
        dropout_prob: Dropout probability applied after embedding lookup and
            before the final linear layer.
        freeze_emb: If ``True``, the embedding weights are not updated during training.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        emb_table=None,
        dropout_prob: float = 0.3,
        freeze_emb: bool = False,
    ):
        """Initialises the classifier network layers.

        Args:
            vocab_size: Vocabulary size including ``<pad>`` (index 0) and ``<unk>``.
            embedding_dim: Dimensionality of token embeddings.
            hidden_dim: Hidden size per GRU direction.
            num_classes: Number of output classes.
            emb_table: Optional numpy array of shape ``(vocab_size, embedding_dim)``
                used to initialise the embedding layer. If ``None``, embeddings are
                randomly initialised.
            dropout_prob: Dropout probability applied after embedding lookup and
                before the final linear layer.
            freeze_emb: If ``True``, the embedding weights are not updated during training.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        if emb_table is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(emb_table))
        self.word_embeddings.weight.requires_grad = not freeze_emb

        self.gru = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        # 2 × hidden_dim: forward + backward final hidden states concatenated
        self.linear = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Compute class logits for a batch of padded token sequences.

        Args:
            x: Either a Long tensor of shape ``(batch_size, max_seq_len)`` for
                vocabulary-index input (static embeddings), or a Float tensor of
                shape ``(batch_size, max_seq_len, embedding_dim)`` containing
                pre-computed contextual embeddings (e.g. from BERT). The dtype
                is used to distinguish the two cases.
            lengths: Optional long tensor of shape ``(batch_size,)`` with the
                unpadded length of each sequence. When provided, packed sequences
                are used so the GRU skips padding tokens.

        Returns:
            Logit tensor of shape ``(batch_size, num_classes)``.
        """
        # Float input = pre-computed contextual embeddings; skip the lookup.
        if x.dtype == torch.long:
            embeds = self.dropout(self.word_embeddings(x))
        else:
            embeds = self.dropout(x)

        if lengths is not None:
            packed = pack_padded_sequence(
                embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(embeds)

        # hidden: (num_directions, batch, hidden_dim) for a 1-layer bidirectional GRU
        combined = self.dropout(
            torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)
        )
        return self.linear(combined)


def build_bigru_dataloader(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    energy_labels: torch.Tensor,
    damage_labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap padded sequences and labels in a DataLoader for the shared training loop.

    Produces batches in the ``(D, DL, Energy, Risk)`` format expected by the
    shared training loop, where:

    - ``D`` = padded token index sequences
    - ``DL`` = unpadded sequence lengths (consumed when ``need_length=True``)
    - ``Energy`` = energy-type class labels (used when ``energy_model=True``)
    - ``Risk`` = potential-damage class labels (used when ``energy_model=False``)

    Args:
        sequences: Long tensor of shape ``(num_samples, max_seq_len)`` with
            padded token indices.
        lengths: Long tensor of shape ``(num_samples,)`` with the unpadded
            length of each sequence.
        energy_labels: Long tensor of shape ``(num_samples,)`` with encoded
            energy-type class indices.
        damage_labels: Long tensor of shape ``(num_samples,)`` with encoded
            potential-damage class indices.
        batch_size: Samples per batch. Defaults to 32.
        shuffle: Whether to shuffle each epoch. Defaults to ``True``.

    Returns:
        A DataLoader yielding 4-tuples of
        ``(sequences, lengths, energy_labels, damage_labels)``.
    """
    dataset = TensorDataset(sequences, lengths, energy_labels, damage_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
