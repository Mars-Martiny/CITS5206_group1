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

