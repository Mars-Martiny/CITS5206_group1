"""BERT transformer embedding backend implementation."""

import torch
import torch.nn as nn
from transformers import AutoModel

from .base import BaseEmbeddingBackend
from .bert_config import BertEmbeddingConfig
from .embedding_output import EmbeddingOutput


class BertEmbeddingBackend(nn.Module, BaseEmbeddingBackend):
    """BERT-based contextual embedding backend.

    This backend consumes already-tokenized transformer inputs such as
    ``input_ids`` and ``attention_mask`` and returns contextual token
    embeddings in the shared :class:`EmbeddingOutput` format.

    The underlying transformer backbone is loaded using
    ``AutoModel.from_pretrained(...)``.

    :param config: BERT embedding configuration.
    :type config: BertEmbeddingConfig
    """

    def __init__(self, config: BertEmbeddingConfig) -> None:
        """Load the transformer and apply configuration (freeze if requested)."""
        super().__init__()
        config.validate()
        self.config = config

        self.model = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)

        if not config.fine_tune:
            self.freeze()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> EmbeddingOutput:
        """Run transformer encoding and return standardized output.

        :param input_ids: Transformer token ids of shape
            ``[batch_size, sequence_length]``.
        :type input_ids: torch.Tensor
        :param attention_mask: Transformer attention mask of shape
            ``[batch_size, sequence_length]``.
        :type attention_mask: torch.Tensor
        :param token_type_ids: Optional segment ids for models that use them.
        :type token_type_ids: torch.Tensor | None
        :returns: Standardized embedding output.
        :rtype: EmbeddingOutput
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(**model_inputs)

        # Base AutoModel outputs last_hidden_state for encoder models such as BERT
        token_embeddings = outputs.last_hidden_state
        # Optional embedding-level dropout before sentence pooling.
        # This is separate from classifier-head dropout.
        token_embeddings = self.dropout(token_embeddings)

        if self.config.pooling == "cls":
            sentence_embedding = token_embeddings[:, 0, :]
        else:
            sentence_embedding = self._masked_mean_pool(
                token_embeddings=token_embeddings,
                attention_mask=attention_mask,
            )

        return EmbeddingOutput(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask.long(),
            sentence_embedding=sentence_embedding,
        )

    def _masked_mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked mean pooling over transformer token embeddings.

        :param token_embeddings: Transformer hidden states of shape
            ``[batch_size, sequence_length, hidden_dim]``.
        :type token_embeddings: torch.Tensor
        :param attention_mask: Attention mask of shape
            ``[batch_size, sequence_length]``.
        :type attention_mask: torch.Tensor
        :returns: Mean-pooled sentence embeddings of shape
            ``[batch_size, hidden_dim]``.
        :rtype: torch.Tensor
        """
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def get_output_dim(self) -> int:
        """Return transformer hidden size."""
        return int(self.model.config.hidden_size)

    def freeze(self) -> None:
        """Freeze transformer parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze transformer parameters."""
        for param in self.model.parameters():
            param.requires_grad = True