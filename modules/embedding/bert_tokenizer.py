"""Hugging Face tokenizer wrapper for BERT embedding inputs."""

from typing import Sequence

import torch
from transformers import AutoTokenizer

from .bert_config import BertEmbeddingConfig


class BertTokenizerWrapper:
    """Tokenizer wrapper for transformer-based embedding backends.

    This class isolates Hugging Face tokenization from the rest of the pipeline.
    It converts raw text strings into tensors suitable for
    transformer models.

    :param config: BERT embedding configuration.
    :type config: BertEmbeddingConfig
    """

    def __init__(self, config: BertEmbeddingConfig) -> None:
        """Load the tokenizer for ``config.model_name``."""
        config.validate()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def encode_texts(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts for transformer input.

        :param texts: Raw or lightly normalized text strings.
        :type texts: Sequence[str]
        :returns: Dictionary containing at least:
            - ``input_ids``: ``[batch_size, sequence_length]``
            - ``attention_mask``: ``[batch_size, sequence_length]``
            and possibly other model-specific fields such as
            ``token_type_ids``.
        :rtype: dict[str, torch.Tensor]
        """
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )