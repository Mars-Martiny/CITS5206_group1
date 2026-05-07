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
#     display_name: nlp
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
## Need to install contractions. Need to list this!
# # %pip install contractions

# %% [markdown]
# # Search for TF-IDF Potential Damage Model

# %%
from modules import OneTextPreProcessor

import json

# Open and read the file
with open('column_map.json', 'r') as file:
    column_map = json.load(file)

# %%
train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

# %%
# If wanted to use spacy transformer model, set to en_core_web_trf (better result, at 11x slower tradeoff)
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

# %%
import torch
from experiment_setup.tf_idf_runner import tf_idf_hparam_search, tf_idf_run_multiple


train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

text_col = column_map["Detailed Description of Event"]         # "description"
label_col = "energy_type"  # or "Potential Damage" for model2

# hparam_study = tf_idf_hparam_search(
#     train_df, valid_df, test_df, text_col,
#     keep_numbers=False, lemma_config=lemma_config,
#     energy_model=False,
#     n_trials=40,
#     timeout=None,   # set to seconds e.g. 3600 to cap wall time
# )

# print("Best val f1_macro :", hparam_study.best_value)
# print("Best params       :", hparam_study.best_params)
# # Visualise results — requires plotly: pip install plotly
# import optuna

# optuna.visualization.plot_optimization_history(hparam_study).show()
# optuna.visualization.plot_param_importances(hparam_study).show()
# optuna.visualization.plot_parallel_coordinate(hparam_study).show()
# # Retrain n times with the best found params
# best = hparam_study.best_params
# _BEST_EPOCHS = best["epochs"]

# best_train_config = {
#     "epochs": _BEST_EPOCHS,
#     "hidden_dim": best["hidden_dim"],
#     "patience": 15,
#     "best_metric": "f1_macro",
#     "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=best["lr"]),
#     "scheduler_fn": lambda opt: (
#         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_BEST_EPOCHS, eta_min=1e-6)
#         if best["scheduler"] == "cosine"
#         else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, _BEST_EPOCHS // 5), gamma=0.5)
#         if best["scheduler"] == "step"
#         else None
#     ),
#     "scheduler_step_per_batch": False,
# }

# best_models = tf_idf_run_multiple(
#     train_df, valid_df, test_df, text_col,
#     keep_numbers=False, lemma_config=lemma_config,
#     energy_model=False, n=5,
#     train_config=best_train_config,
#     requirements={
#         "confidence_threshold": {"high": 0.80, "medium": 0.50},
#         "high_threshold": 0.70,
#         "fatal_accuracy": 0.95,
#         "f1_target": {5: 0.70, 6: 0.70, 11: 0.70, 17: 0.70, 0:0.0}, # Class 0 (Animal) no target
#     }
# )

# %% [markdown]
# ## New Section

# %%
from modules import OneTextPreProcessor
import pandas as pd 

def pre_process(data_path):
    oneTextPreProcessor = OneTextPreProcessor(keep_numbers=False, column_map=column_map)
    proc_df = oneTextPreProcessor.pre_process_df(
        pd.read_csv(data_path),
        column_map["Detailed Description of Event"]
    )
    return proc_df

model1_train = pre_process("dataset/model1_train.csv")
model1_valid = pre_process("dataset/model1_valid.csv")
model1_test = pre_process("dataset/model1_test.csv")

# %%
import torch
from implementations.simple_bi_gru import BiGRUClassifier, build_bigru_dataloader
from modules.training_loop import training
from modules.encoding import LabelEncoder
from modules.encoding.vocab_encoder import VocabEncoder
from modules.encoding.sequence_encoder import SequenceEncoder

tokens_col = "description_tokens_lemma"

train_tokenized_docs = model1_train[tokens_col].tolist()
val_tokenized_docs   = model1_valid[tokens_col].tolist()
test_tokenized_docs  = model1_test[tokens_col].tolist()


# %%
# Build vocab on training set only (min_freq=2 filters noise)
vocab_enc = VocabEncoder(min_freq=2)
vocab_enc.fit(train_tokenized_docs)

# Determine max_len from training set (95th percentile avoids outlier padding)
import numpy as np
train_lens_raw = [len(doc) for doc in train_tokenized_docs]
max_len = int(np.percentile(train_lens_raw, 95))
print(f"vocab_size={vocab_enc.vocab_size}, max_len={max_len}")

seq_enc = SequenceEncoder(vocab_enc, max_length=max_len)

def encode_split(docs):
    seqs    = torch.tensor(seq_enc.encode_sequences(docs), dtype=torch.long)
    lengths = torch.tensor([min(len(d), max_len) for d in docs], dtype=torch.long)
    return seqs, lengths

train_seqs, train_lengths = encode_split(train_tokenized_docs)
val_seqs,   val_lengths   = encode_split(val_tokenized_docs)
test_seqs,  test_lengths  = encode_split(test_tokenized_docs)


# %%
# Energy type labels
energy_enc = LabelEncoder()
energy_enc.fit(model1_train["energy_type"].tolist())
train_energy = torch.tensor(energy_enc.encode_many(model1_train["energy_type"].tolist()))
val_energy   = torch.tensor(energy_enc.encode_many(model1_valid["energy_type"].tolist()))
test_energy  = torch.tensor(energy_enc.encode_many(model1_test["energy_type"].tolist()))

# Potential damage labels
damage_enc = LabelEncoder()
damage_enc.fit(model1_train["potential_damage"].tolist())
train_damage = torch.tensor(damage_enc.encode_many(model1_train["potential_damage"].tolist()))
val_damage   = torch.tensor(damage_enc.encode_many(model1_valid["potential_damage"].tolist()))
test_damage  = torch.tensor(damage_enc.encode_many(model1_test["potential_damage"].tolist()))


# %%
train_dl = build_bigru_dataloader(train_seqs, train_lengths, train_energy, train_damage)
val_dl   = build_bigru_dataloader(val_seqs,   val_lengths,   val_energy,   val_damage,   shuffle=False)
test_dl  = build_bigru_dataloader(test_seqs,  test_lengths,  test_energy,  test_damage,  shuffle=False)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = energy_enc.num_classes

model = BiGRUClassifier(
    vocab_size=vocab_enc.vocab_size,
    embedding_dim=128,
    hidden_dim=128,
    num_classes=num_classes,
).to(device)

results = training(
    model_type="bigru",
    model=model,
    train_dl=train_dl,
    valid_dl=val_dl,
    test_dl=test_dl,
    epochs=50,
    device=device,
    patience=10,
    best_metric="f1_macro",
    criterion_type="focal",
    need_length=True,   # tells the loop to call model(D, DL)
    energy_model=True,  # set False to predict potential_damage instead
    num_classes=num_classes,
)


# %%
from modules.embedding.safety_bert_static import get_safety_bert_embedding_matrix


device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = energy_enc.num_classes

model = BiGRUClassifier(
    vocab_size=vocab_enc.vocab_size,
    embedding_dim=128,
    hidden_dim=128,
    num_classes=num_classes,
).to(device)

results = training(
    model_type="bigru",
    model=model,
    train_dl=train_dl,
    valid_dl=val_dl,
    test_dl=test_dl,
    epochs=50,
    device=device,
    patience=10,
    best_metric="f1_macro",
    criterion_type="focal",
    need_length=True,   # tells the loop to call model(D, DL)
    energy_model=False,  # set False to predict potential_damage instead
    num_classes=num_classes,
)

# vocab = {
#     "worker": 0,
#     "fell": 1,
#     "on": 2,
#     "ladder": 3
# }
# matrix = get_safety_bert_embedding_matrix(vocab)
# print(matrix.shape)  # → torch.Size([4, 768])
# print(matrix)

# %%
from modules.embedding.safety_bert_static import get_safety_bert_embedding_matrix

# vocab_enc.vocab is the {word: index} dict your tokenizer already built
bert_matrix = get_safety_bert_embedding_matrix(
    vocab=vocab_enc.token_to_id,          # dict[str, int]
    model_name="bert-base-uncased", # or your SafetyBERT checkpoint
    device=device,
    verbose=True,
)


# bert-base-uncased has hidden_dim=768, so embedding_dim must match
model = BiGRUClassifier(
    vocab_size=vocab_enc.vocab_size,
    embedding_dim=768,              # must match BERT hidden dim
    hidden_dim=128,
    num_classes=num_classes,
    emb_table=bert_matrix.cpu().numpy(),          # inject the BERT vectors
    freeze_emb=False,               # True = frozen, False = fine-tunable
).to(device)

results = training(
    model_type="bigru_safetybert_static",
    model=model,
    train_dl=train_dl,
    valid_dl=val_dl,
    test_dl=test_dl,
    epochs=50,
    device=device,
    patience=10,
    best_metric="f1_macro",
    criterion_type="focal",
    need_length=True,   # tells the loop to call model(D, DL)
    energy_model=True,  # set False to predict potential_damage instead
    num_classes=num_classes,
)


# %%
from torch.nn.utils.rnn import pad_sequence
from modules.embedding.safety_bert_static import get_contextual_embeddings

# ── 1. Encode all splits (raw text, before tokenizer) ──────────────────
train_embs = get_contextual_embeddings(model1_train["description"], model_name="bert-base-uncased", device=device)
val_embs   = get_contextual_embeddings(model1_valid["description"],   model_name="bert-base-uncased", device=device)
test_embs  = get_contextual_embeddings(model1_test["description"],  model_name="bert-base-uncased", device=device)

# ── 2. Pad to uniform length for batching ───────────────────────────────────
def pack(embs):
    lengths = torch.tensor([e.shape[0] for e in embs])
    padded  = pad_sequence(embs, batch_first=True)   # (N, max_seq_len, 768)
    return padded, lengths

train_seq, train_len = pack(train_embs)
val_seq,   val_len   = pack(val_embs)
test_seq,  test_len  = pack(test_embs)

# ── 3. Build DataLoaders ──────────────────────────────────────────────────
train_dl = build_bigru_dataloader(train_seq, train_len, train_energy, train_damage)
val_dl   = build_bigru_dataloader(val_seq,   val_len,   val_energy,   val_damage,   shuffle=False)
test_dl  = build_bigru_dataloader(test_seq,  test_len,  test_energy,  test_damage,  shuffle=False)

# ── 4. Model — vocab_size is a placeholder, embedding layer is bypassed ─────
model = BiGRUClassifier(
    vocab_size=1,          # unused — forward() detects float input and skips it
    embedding_dim=768,     # must match BERT hidden dim
    hidden_dim=128,
    num_classes=num_classes,
    emb_table=None,        # unused
    freeze_emb=False,
).to(device)

# ── 5. Training — identical call ─────────────────────────────────────────────
results = training(
    model_type="bigru_safetybert_contextual",
    model=model,
    train_dl=train_dl,
    valid_dl=val_dl,
    test_dl=test_dl,
    epochs=50,
    device=device,
    patience=10,
    best_metric="f1_macro",
    criterion_type="focal",
    need_length=True,
    energy_model=True,
    num_classes=num_classes,
)

