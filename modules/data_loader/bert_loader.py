"""DataLoader utilities for BERT-based text classification."""

import torch
from torch.utils.data import Dataset, DataLoader


class BertTextDataset(Dataset):
    """Dataset for BERT text classification.
    
    returns tokenized inputs and labels for each text sample.
    """

    def __init__(self, texts, labels, tokenizer_wrapper):
        """Initialize the dataset with raw texts, labels, and a tokenizer wrapper.
        
        :param texts: List of raw text samples.
        :type texts: list[str]
        :param labels: List of corresponding label ids.
        :type labels: list[int]
        :param tokenizer_wrapper: Tokenizer wrapper for encoding text.
        :type tokenizer_wrapper: BertTokenizerWrapper
        """
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer_wrapper = tokenizer_wrapper

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Return the tokenized inputs and label for the sample at index `idx`."""
        encoded = self.tokenizer_wrapper.encode_texts([self.texts[idx]])

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item


def df_to_bert_dataloader(
    df,
    text_col: str,
    label_col: str,
    tokenizer_wrapper,
    batch_size: int,
    shuffle: bool = True,
):
    """Convert dataframe text and labels into a BERT-compatible DataLoader.
    
    :param df: Input dataframe containing text and label columns.
    :type df: pandas.DataFrame
    :param text_col: Name of the column containing text data.
    :type text_col: str
    :param label_col: Name of the column containing label data.
    :type label_col: str
    :param tokenizer_wrapper: Tokenizer wrapper for encoding text.
    :type tokenizer_wrapper: BertTokenizerWrapper
    :param batch_size: Batch size for the DataLoader.
    :type batch_size: int
    :param shuffle: Whether to shuffle the data each epoch.
    :type shuffle: bool
    
    :returns: DataLoader yielding batches of tokenized text and labels.
    :rtype: torch.utils.data.DataLoader 
    """
    dataset = BertTextDataset(
        texts=df[text_col].fillna("").astype(str).tolist(),
        labels=df[label_col].tolist(),
        tokenizer_wrapper=tokenizer_wrapper,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)