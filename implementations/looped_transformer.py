"""Looped (weight-tied) transformer classifier for single-text incident classification."""

import torch
import torch.nn as nn

class LoopedTransformerBlock(nn.Module):
    """Single shared block that gets reused across depth iterations.

    Implements a pre-norm transformer encoder block (multi-head self-attention
    followed by a position-wise feed-forward network with GELU activation) that
    is intended to be applied repeatedly to its own output, sharing weights
    across depth.

    Args:
        d_model: Hidden size of the model and the attention/FFN sublayers.
        nhead: Number of attention heads. Must divide ``d_model``.
        dim_feedforward: Hidden size of the position-wise feed-forward network.
        dropout: Dropout probability applied inside attention, the FFN, and on
            each residual branch.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """Initialises the attention, feed-forward, and normalisation layers.

        Args:
            d_model: Hidden size of the model and the attention/FFN sublayers.
            nhead: Number of attention heads. Must divide ``d_model``.
            dim_feedforward: Hidden size of the position-wise feed-forward network.
            dropout: Dropout probability applied inside attention, the FFN, and
                on each residual branch.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        """Apply one pre-norm self-attention + feed-forward iteration.

        Args:
            x: Float tensor of shape ``(batch_size, seq_len, d_model)`` with the
                current token representations.
            src_key_padding_mask: Optional bool tensor of shape
                ``(batch_size, seq_len)`` where ``True`` marks padding positions
                that should be ignored by attention.

        Returns:
            Float tensor of shape ``(batch_size, seq_len, d_model)`` with the
            updated token representations after one block iteration.
        """
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class LoopedTransformer(nn.Module):
    """Weight-tied transformer classifier with depth implemented as iteration.

    Embeds token indices, adds learned positional embeddings, then applies a
    single shared :class:`LoopedTransformerBlock` ``num_loops`` times before
    pooling the first (CLS-style) token and projecting to class logits. Sharing
    weights across depth keeps the parameter count constant regardless of how
    many loops are used.

    Args:
        vocab_size: Vocabulary size including ``<pad>`` (index 0).
        d_model: Hidden size of the model and the shared transformer block.
        nhead: Number of attention heads. Must divide ``d_model``.
        dim_feedforward: Hidden size of the position-wise feed-forward network.
        num_loops: Number of times the shared block is applied to the input.
        num_classes: Number of output classes.
        max_seq_len: Maximum supported sequence length for the learned
            positional embedding table.
        dropout: Dropout probability applied to embeddings and inside the
            shared transformer block.
    """

    def __init__(self, vocab_size, d_model, nhead, dim_feedforward,
                 num_loops, num_classes, max_seq_len=512, dropout=0.1):
        """Initialises the embeddings, shared block, and classifier head.

        Args:
            vocab_size: Vocabulary size including ``<pad>`` (index 0).
            d_model: Hidden size of the model and the shared transformer block.
            nhead: Number of attention heads. Must divide ``d_model``.
            dim_feedforward: Hidden size of the position-wise feed-forward network.
            num_loops: Number of times the shared block is applied to the input.
            num_classes: Number of output classes.
            max_seq_len: Maximum supported sequence length for the learned
                positional embedding table.
            dropout: Dropout probability applied to embeddings and inside the
                shared transformer block.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.block = LoopedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
        self.num_loops = num_loops
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Compute class logits for a batch of padded token sequences.

        Args:
            input_ids: Long tensor of shape ``(batch_size, seq_len)`` with
                vocabulary indices for each token.
            attention_mask: Optional long/bool tensor of shape
                ``(batch_size, seq_len)`` in HuggingFace style, where ``1``
                marks real tokens and ``0`` marks padding. Internally converted
                to PyTorch's ``key_padding_mask`` (``True`` = ignore).
            token_type_ids: Unused, accepted only for interface compatibility
                with HuggingFace-style batches.

        Returns:
            Logit tensor of shape ``(batch_size, num_classes)`` produced from
            the first (CLS-style) token after ``num_loops`` block iterations.
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embedding(input_ids) + self.pos_embedding(positions))

        # Convert HF-style attention mask (1=attend, 0=ignore)
        # to PyTorch's key_padding_mask (True=ignore)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None

        for _ in range(self.num_loops):
            x = self.block(x, src_key_padding_mask=pad_mask)

        x = self.norm(x)
        # CLS-style pooling: use [0] token, or mean pool — your call
        pooled = x[:, 0, :]  # or x.mean(dim=1)
        return self.classifier(pooled)