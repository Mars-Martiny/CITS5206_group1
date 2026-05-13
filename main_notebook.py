# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: torch-rocm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # General Common Config

# %%
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

import json

# Open and read the file
with open('column_map.json', 'r') as file:
    column_map = json.load(file)

text_col = column_map["Detailed Description of Event"]  

# %% [markdown]
# ## Energy Model

# %%
from experiment_setup.bi_gru_runner import bigru_run_single
import pandas as pd

train_df = pd.read_csv("dataset/model1_train.csv")
valid_df = pd.read_csv("dataset/model1_valid.csv")
test_df = pd.read_csv("dataset/model1_test.csv")

result = bigru_run_single(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    lemma_config=lemma_config,
    train_config={"embedding_type": "none", "epochs": 50},
)

# For SafetyBERT static:
result = bigru_run_single(train_df, valid_df, test_df, text_col,
    energy_model=True,
    lemma_config=lemma_config,
    train_config={"embedding_type": "static", "embedding_model_name": "adanish91/safetybert"})

# For contextual BERT:
result = bigru_run_single(train_df, valid_df, test_df, text_col,
    energy_model=True,
    lemma_config=lemma_config,
    train_config={"embedding_type": "contextual", "embedding_model_name": "bert-base-uncased"})


# %% [markdown]
# ## Damage Model

# %%
from experiment_setup.bi_gru_runner import bigru_run_single

train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

result = bigru_run_single(
    train_df, valid_df, test_df, text_col,
    energy_model=False,
    lemma_config=lemma_config,
    train_config={"embedding_type": "none", "epochs": 50},
)

# For SafetyBERT static:
result = bigru_run_single(train_df, valid_df, test_df, text_col,
    energy_model=False,
    lemma_config=lemma_config,
    train_config={"embedding_type": "static", "embedding_model_name": "adanish91/safetybert"})

# For contextual BERT:
result = bigru_run_single(train_df, valid_df, test_df, text_col,
    energy_model=False,
    lemma_config=lemma_config,
    train_config={"embedding_type": "contextual", "embedding_model_name": "bert-base-uncased"})


# %% [markdown]
# ## Looped Transformer — Energy Model

# %%
import torch
import pandas as pd

from modules.data_loader.bert_loader import df_to_bert_dataloader
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_tokenizer import BertTokenizerWrapper
from modules.encoding import LabelEncoder
from modules.training_loop import training
from implementations.looped_transformer import LoopedTransformer

train_df = pd.read_csv("dataset/model1_train.csv")
valid_df = pd.read_csv("dataset/model1_valid.csv")
test_df  = pd.read_csv("dataset/model1_test.csv")

label_enc = LabelEncoder()
label_enc.fit(train_df["Energy Type"].tolist())
for df in (train_df, valid_df, test_df):
    df["_label"] = label_enc.encode_many(df["Energy Type"].tolist())

import numpy as np
from transformers import AutoTokenizer as _AutoTokenizer

_raw_tok = _AutoTokenizer.from_pretrained("adanish91/safetybert")
_lengths = [len(_raw_tok(t, truncation=False)["input_ids"]) for t in train_df["Detailed Description of Event"]]
_max_length = int(np.percentile(_lengths, 95))

tokenizer = BertTokenizerWrapper(BertEmbeddingConfig(model_name="adanish91/safetybert", max_length=_max_length))

train_dl = df_to_bert_dataloader(train_df, "Detailed Description of Event", "_label", tokenizer, batch_size=32)
valid_dl = df_to_bert_dataloader(valid_df, "Detailed Description of Event", "_label", tokenizer, batch_size=64, shuffle=False)
test_dl  = df_to_bert_dataloader(test_df,  "Detailed Description of Event", "_label", tokenizer, batch_size=64, shuffle=False)

cfg = {
    "scheduler_step_per_batch": False,
    "best_metric": "f1_macro",
    "save": True,
    "log_leaderboard": True,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LoopedTransformer(
    vocab_size=30522,
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    num_loops=6,
    num_classes=label_enc.num_classes,
).to(device)

result = training(
    model=model,
    model_type="looped_transformer",
    energy_model=True,
    need_length=False,
    train_dl=train_dl,
    valid_dl=valid_dl,
    test_dl=test_dl,
    num_classes=label_enc.num_classes,
    class_dict=label_enc.id_to_label,
    epochs=50,
    patience=10,
    criterion_type="focal",
    device=device,
    **cfg
)


# %% [markdown]
# ## Looped Transformer — Damage Model

# %%
train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df  = pd.read_csv("dataset/model2_test.csv")

label_enc2 = LabelEncoder()
label_enc2.fit(train_df["Type of Potential Damage"].tolist())
for df in (train_df, valid_df, test_df):
    df["_label"] = label_enc2.encode_many(df["Type of Potential Damage"].tolist())

train_dl2 = df_to_bert_dataloader(train_df, "Detailed Description of Event", "_label", tokenizer, batch_size=32)
valid_dl2 = df_to_bert_dataloader(valid_df, "Detailed Description of Event", "_label", tokenizer, batch_size=64, shuffle=False)
test_dl2  = df_to_bert_dataloader(test_df,  "Detailed Description of Event", "_label", tokenizer, batch_size=64, shuffle=False)

cfg = {
    "scheduler_step_per_batch": False,
    "best_metric": "f1_macro",
    "save": True,
    "log_leaderboard": True,
}

model2 = LoopedTransformer(
    vocab_size=30522,
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    num_loops=6,
    num_classes=label_enc2.num_classes,
).to(device)

result2 = training(
    model=model2,
    model_type="looped_transformer",
    energy_model=False,
    need_length=False,
    train_dl=train_dl2,
    valid_dl=valid_dl2,
    test_dl=test_dl2,
    num_classes=label_enc2.num_classes,
    class_dict=label_enc2.id_to_label,
    epochs=50,
    patience=10,
    criterion_type="focal",
    device=device,
    **cfg
)
