"""Configuration dataclass for BERT-based embedding backends."""

from dataclasses import dataclass


@dataclass
class BertEmbeddingConfig:
    """Configuration for a BERT-based embedding backend.

    :param model_name: Hugging Face model identifier, such as
        ``"bert-base-uncased"``.
    :type model_name: str
    :param max_length: Maximum sequence length for tokenizer output.
    :type max_length: int
    :param dropout: Embedding dropout applied to transformer token embeddings
        before pooling. This is separate from classifier-head dropout.
    :type dropout: float
    :param pooling: Pooling strategy for sentence embedding. Supported values:
        - ``"cls"``
        - ``"mean"``
    :type pooling: str
    :param fine_tune: Whether transformer weights remain trainable.
    :type fine_tune: bool
    """

    model_name: str = "bert-base-uncased"
    tokenizer_name: str | None = None
    max_length: int = 160
    dropout: float = 0.1
    pooling: str = "mean"
    fine_tune: bool = True

    def validate(self) -> None:
        """Validate configuration values.

        :raises ValueError: If any field is invalid.
        """
        if not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if self.tokenizer_name is not None and not self.tokenizer_name:
            raise ValueError("tokenizer_name must be None or a non-empty string")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0)")
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")