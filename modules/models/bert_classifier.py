"""BERT classifier for transformer-based incident classification."""

import torch.nn as nn

from modules.embedding.bert_embedding import BertEmbeddingBackend


class BertClassifier(nn.Module):
    """BERT embedding backend followed by a classification head.
    
    The classifier consists of a dropout layer and a linear layer mapping from the embedding dimension to the number of classes.
    The forward pass takes tokenized inputs (input_ids, attention_mask, token_type_ids), obtains the sentence embedding from the embedding backend, and applies the classifier to produce class logits.
    This model is designed for text classification tasks where the BERT embedding backend provides rich contextual representations of the input text, and the classifier head maps these representations to the target classes.
    The modular design allows for easy swapping of the embedding backend or modification of the classifier architecture as needed for different tasks or datasets.
    """

    def __init__(
        self,
        embedding_backend: BertEmbeddingBackend,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the BERT classifier with the given embedding backend, number of classes, and dropout rate.
        
        :param embedding_backend: Pre-trained BERT embedding backend to use for obtaining sentence embeddings.
        :type embedding_backend: BertEmbeddingBackend
        :param num_classes: Number of target classes for classification.
        :type num_classes: int
        :param dropout: Dropout rate for the classifier.
        :type dropout: float
        """
        super().__init__()
        self.embedding_backend = embedding_backend
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_backend.get_output_dim(), num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Compute class logits for a batch of tokenized inputs.
        
        :param input_ids: Tensor of token IDs.
        :type input_ids: torch.Tensor
        :param attention_mask: Tensor of attention masks.
        :type attention_mask: torch.Tensor
        :param token_type_ids: Tensor of token type IDs.
        :type token_type_ids: torch.Tensor, optional
        :returns: Class logits.
        :rtype: torch.Tensor
        """
        embedding_output = self.embedding_backend(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(embedding_output.sentence_embedding)