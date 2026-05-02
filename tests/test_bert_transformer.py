"""Unit tests for BERT transformer implementation.

These tests avoid downloading real Hugging Face models by mocking the BERT
tokenizer and embedding backend. They validate that the BERT pipeline wiring
works correctly before running expensive real experiments.
"""

import pytest
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from modules.encoding.label_encoder import LabelEncoder
from modules.data_loader.bert_loader import BertTextDataset, df_to_bert_dataloader
from modules.models.bert_classifier import BertClassifier
from implementations.bert_transformer import encode_label_column


# ─────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────

class DummyTokenizerWrapper:
    """Fake tokenizer wrapper returning fixed-size BERT-like tensors."""

    def __init__(self, max_length=8):
        self.max_length = max_length

    def encode_texts(self, texts):
        batch_size = len(texts)
        return {
            "input_ids": torch.ones(batch_size, self.max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, self.max_length, dtype=torch.long),
            "token_type_ids": torch.zeros(batch_size, self.max_length, dtype=torch.long),
        }


class DummyEmbeddingBackend(nn.Module):
    """Fake embedding backend compatible with BertClassifier."""

    def __init__(self, output_dim=16):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(output_dim, output_dim)

    def get_output_dim(self):
        return self.output_dim

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        batch_size = input_ids.size(0)
        sentence_embedding = torch.ones(batch_size, self.output_dim)

        class Output:
            pass

        output = Output()
        output.sentence_embedding = sentence_embedding
        return output


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "description": [
                "Worker slipped near conveyor",
                "Vehicle reversed into bollard",
                "Chemical spill near workshop",
                "Object fell from height",
            ],
            "energy_type": [
                "Gravitational",
                "Vehicular",
                "Chemical",
                "Gravitational",
            ],
        }
    )


@pytest.fixture
def split_dfs(sample_df):
    train_df = sample_df.copy()
    valid_df = sample_df.copy()
    test_df = sample_df.copy()
    return train_df, valid_df, test_df


# ─────────────────────────────────────────────────────────────
# Label encoding tests
# ─────────────────────────────────────────────────────────────

def test_encode_label_column_encodes_all_splits(split_dfs):
    train_df, valid_df, test_df = split_dfs

    train_enc, valid_enc, test_enc, label_encoder, class_names = encode_label_column(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_col="energy_type",
    )

    assert train_enc["energy_type"].dtype.kind in {"i", "u"}
    assert valid_enc["energy_type"].dtype.kind in {"i", "u"}
    assert test_enc["energy_type"].dtype.kind in {"i", "u"}

    assert label_encoder.num_classes == 3
    assert set(class_names) == {"Chemical", "Gravitational", "Vehicular"}


def test_encode_label_column_raises_for_unknown_validation_label(sample_df):
    train_df = sample_df.copy()
    valid_df = sample_df.copy()
    test_df = sample_df.copy()

    valid_df.loc[0, "energy_type"] = "Unknown Class"

    with pytest.raises(ValueError, match="Unknown label"):
        encode_label_column(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            label_col="energy_type",
        )


def test_encode_label_column_does_not_mutate_original_dataframe(split_dfs):
    train_df, valid_df, test_df = split_dfs
    original_labels = train_df["energy_type"].copy()

    encode_label_column(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_col="energy_type",
    )

    assert train_df["energy_type"].equals(original_labels)


# ─────────────────────────────────────────────────────────────
# BERT dataset / dataloader tests
# ─────────────────────────────────────────────────────────────

def test_bert_text_dataset_returns_expected_keys(sample_df):
    tokenizer = DummyTokenizerWrapper(max_length=8)
    labels = [0, 1, 2, 0]

    dataset = BertTextDataset(
        texts=sample_df["description"].tolist(),
        labels=labels,
        tokenizer_wrapper=tokenizer,
    )

    item = dataset[0]

    assert set(item.keys()) == {
        "input_ids",
        "attention_mask",
        "label",
        "token_type_ids",
    }
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["token_type_ids"].shape == (8,)
    assert item["label"].dtype == torch.long


def test_bert_dataloader_batches_have_correct_shapes(sample_df):
    tokenizer = DummyTokenizerWrapper(max_length=8)

    df = sample_df.copy()
    df["energy_type"] = [0, 1, 2, 0]

    dl = df_to_bert_dataloader(
        df=df,
        text_col="description",
        label_col="energy_type",
        tokenizer_wrapper=tokenizer,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dl))

    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["token_type_ids"].shape == (2, 8)
    assert batch["label"].shape == (2,)


def test_bert_dataloader_handles_missing_text_values(sample_df):
    tokenizer = DummyTokenizerWrapper(max_length=8)

    df = sample_df.copy()
    df.loc[0, "description"] = None
    df["energy_type"] = [0, 1, 2, 0]

    dl = df_to_bert_dataloader(
        df=df,
        text_col="description",
        label_col="energy_type",
        tokenizer_wrapper=tokenizer,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dl))

    assert batch["input_ids"].shape == (2, 8)


# ─────────────────────────────────────────────────────────────
# BERT classifier tests
# ─────────────────────────────────────────────────────────────

def test_bert_classifier_forward_shape():
    embedding_backend = DummyEmbeddingBackend(output_dim=16)
    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=4,
        dropout=0.1,
    )

    input_ids = torch.ones(3, 8, dtype=torch.long)
    attention_mask = torch.ones(3, 8, dtype=torch.long)
    token_type_ids = torch.zeros(3, 8, dtype=torch.long)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    assert logits.shape == (3, 4)


def test_bert_classifier_forward_without_token_type_ids():
    embedding_backend = DummyEmbeddingBackend(output_dim=16)
    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=3,
        dropout=0.1,
    )

    input_ids = torch.ones(2, 8, dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert logits.shape == (2, 3)


def test_bert_classifier_outputs_logits_not_probabilities():
    embedding_backend = DummyEmbeddingBackend(output_dim=16)
    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=3,
        dropout=0.1,
    )

    input_ids = torch.ones(2, 8, dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)

    row_sums = logits.sum(dim=1)

    assert not torch.allclose(row_sums, torch.ones_like(row_sums))


# ─────────────────────────────────────────────────────────────
# Integration-style smoke test
# ─────────────────────────────────────────────────────────────

def test_bert_classifier_can_train_one_step(sample_df):
    tokenizer = DummyTokenizerWrapper(max_length=8)

    df = sample_df.copy()
    df["energy_type"] = [0, 1, 2, 0]

    dl = df_to_bert_dataloader(
        df=df,
        text_col="description",
        label_col="energy_type",
        tokenizer_wrapper=tokenizer,
        batch_size=2,
        shuffle=False,
    )

    embedding_backend = DummyEmbeddingBackend(output_dim=16)
    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=3,
        dropout=0.1,
    )

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(dl))

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
    )

    loss = criterion(logits, batch["label"])

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    assert torch.isfinite(loss)