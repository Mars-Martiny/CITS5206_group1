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
# ## Notes
# Please do not that this is residual code.
# I start experimenting and rapidly prototype on .ipynb then after it is fixed and complete only then I move it to external module.
#
# Exercise caution on using these code as references
#

# %%
from modules import OneTextPreProcessor

import json

# Open and read the file
with open('column_map.json', 'r') as file:
    column_map = json.load(file)

# If wanted to use spacy transformer model, set to en_core_web_trf (better result, at 11x slower tradeoff)
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

# %%
import torch
from experiment_setup.tf_idf_runner import tf_idf_run_multiple, tf_idf_hparam_search
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

# model2_train = pre_process("dataset/model2_train.csv")
# model2_valid = pre_process("dataset/model2_valid.csv")
# model2_test = pre_process("dataset/model2_test.csv")

# %%
model1_train["energy_type"].value_counts()

train_df = pd.read_csv("dataset/model1_train.csv")
valid_df = pd.read_csv("dataset/model1_valid.csv")
test_df = pd.read_csv("dataset/model1_test.csv")

text_col = column_map["Detailed Description of Event"]         # "description"

_EPOCHS = 100

# Baseline: plain TF-IDF features (current default)
tfidf_train_config = {
    "epochs": _EPOCHS,
    "patience": 12,
    "best_metric": "f1_macro",
    # Optimizer factory — receives the model after it is built inside the runner
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    # CosineAnnealingLR: lr decays from initial to eta_min over T_max epochs
    "scheduler_fn": lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=_EPOCHS, eta_min=1e-6
    ),
    "scheduler_step_per_batch": False,
}

models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True, n=5,
    train_config=tfidf_train_config,
text_col = column_map["Detailed Description of Event"]         # "description"
label_col = "energy_type"  # or "Potential Damage" for model2

train_tokenized_docs = model1_train[tokens_col].tolist()
val_tokenized_docs   = model1_valid[tokens_col].tolist()
test_tokenized_docs  = model1_test[tokens_col].tolist()

label_enc = LabelEncoder()
label_enc.fit(model1_train[label_col].tolist())

train_labels = torch.tensor(label_enc.encode_many(model1_train[label_col].tolist()))
val_labels   = torch.tensor(label_enc.encode_many(model1_valid[label_col].tolist()))
test_labels  = torch.tensor(label_enc.encode_many(model1_test[label_col].tolist()))

# --- TF-IDF ---
vectorizer = TFIDFVectorizer().fit(train_tokenized_docs)
train_vecs = vectorizer.transform(train_tokenized_docs)
val_vecs   = vectorizer.transform(val_tokenized_docs)
test_vecs  = vectorizer.transform(test_tokenized_docs)

train_dl = build_tfidf_dataloader(train_vecs, train_labels)
val_dl   = build_tfidf_dataloader(val_vecs, val_labels, shuffle=False)
test_dl  = build_tfidf_dataloader(test_vecs, test_labels, shuffle=False)

num_classes = label_enc.num_classes
model = TFIDFClassifier(vocab_size=len(vectorizer.vocab), num_classes=num_classes, hidden_dim=256)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

results = training(
    model_type="tf_idf",
    model=model,
    train_dl=train_dl,
    valid_dl=val_dl,
    test_dl=test_dl,
    epochs=100,
    device=device,
    patience=12,
    criterion_weights=None,
    best_metric="f1_macro",
    need_length=False,
    energy_model=True,
    num_classes=num_classes,
    requirements={
        "confidence_threshold": {"high": 0.80, "medium": 0.50},
        "high_threshold": 0.70,
        "fatal_accuracy": 0.95,
        "f1_target": {5: 0.70, 6: 0.70, 11: 0.70, 17: 0.70, 0:0.0}, # Class 0 (Animal) no target
    }
)

# %%
from sklearn.metrics import classification_report

model = results["config"]["model"]
valid_dl = results["config"]["valid_dl"]
device = results["config"]["device"]

train_df = pd.read_csv("dataset/model1_train.csv")
valid_df = pd.read_csv("dataset/model1_valid.csv")
test_df = pd.read_csv("dataset/model1_test.csv")

text_col = column_map["Detailed Description of Event"]         # "description"

_EPOCHS = 100

# Baseline: plain TF-IDF features (current default)
tfidf_train_config = {
    "epochs": _EPOCHS,
    "patience": 12,
    "best_metric": "f1_macro",
    # Optimizer factory — receives the model after it is built inside the runner
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    # CosineAnnealingLR: lr decays from initial to eta_min over T_max epochs
    "scheduler_fn": lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=_EPOCHS, eta_min=1e-6
    ),
    "scheduler_step_per_batch": False,
}

models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True, n=5,
    train_config=tfidf_train_config,
)

# %%
# models

# %% [markdown]
# ## TF-IDF Hyperparameter Search (Optuna)
# Searches over: `lr`, `hidden_dim`, `epochs`, `scheduler`
# Pre-processing runs once before the study; each trial trains a fresh model.

# %%
hparam_study = tf_idf_hparam_search(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True,
    n_trials=40,
    timeout=None,   # set to seconds e.g. 3600 to cap wall time
)

print("Best val f1_macro :", hparam_study.best_value)
print("Best params       :", hparam_study.best_params)

# %%
# Visualise results — requires plotly: pip install plotly
import optuna

optuna.visualization.plot_optimization_history(hparam_study).show()
optuna.visualization.plot_param_importances(hparam_study).show()
optuna.visualization.plot_parallel_coordinate(hparam_study).show()

# %%
# Retrain n times with the best found params
best = hparam_study.best_params
_BEST_EPOCHS = best["epochs"]

best_train_config = {
    "epochs": _BEST_EPOCHS,
    "hidden_dim": best["hidden_dim"],
    "patience": 15,
    "best_metric": "f1_macro",
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=best["lr"]),
    "scheduler_fn": lambda opt: (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_BEST_EPOCHS, eta_min=1e-6)
        if best["scheduler"] == "cosine"
        else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, _BEST_EPOCHS // 5), gamma=0.5)
        if best["scheduler"] == "step"
        else None
    ),
    "scheduler_step_per_batch": False,
}

best_models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True, n=5,
    train_config=best_train_config,
)

# %% [markdown]
# # Search for TF-IDF Potential Damage Model

# %%
train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

# %%
valid_df["Type of Potential Damage"].value_counts()

# %%
train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

hparam_study = tf_idf_hparam_search(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=False,
    n_trials=40,
    timeout=None,   # set to seconds e.g. 3600 to cap wall time
)

print("Best val f1_macro :", hparam_study.best_value)
print("Best params       :", hparam_study.best_params)
# Visualise results — requires plotly: pip install plotly
import optuna

optuna.visualization.plot_optimization_history(hparam_study).show()
optuna.visualization.plot_param_importances(hparam_study).show()
optuna.visualization.plot_parallel_coordinate(hparam_study).show()
# Retrain n times with the best found params
best = hparam_study.best_params
_BEST_EPOCHS = best["epochs"]

best_train_config = {
    "epochs": _BEST_EPOCHS,
    "hidden_dim": best["hidden_dim"],
    "patience": 15,
    "best_metric": "f1_macro",
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=best["lr"]),
    "scheduler_fn": lambda opt: (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_BEST_EPOCHS, eta_min=1e-6)
        if best["scheduler"] == "cosine"
        else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, _BEST_EPOCHS // 5), gamma=0.5)
        if best["scheduler"] == "step"
        else None
    ),
    "scheduler_step_per_batch": False,
}

best_models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=False, n=5,
    train_config=best_train_config,
)

# %%
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import numpy as np

# %%
# ── Load once, reuse for both extraction methods ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
safety_bert = AutoModel.from_pretrained("adanish91/safetybert")


# %%
# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: Static token embeddings  (for Bi-GRU embedding matrix init)
# Pull the raw embedding weight matrix — shape: (vocab_size=30522, hidden=768)
# This is a direct lookup table, no context.
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding_matrix(vocab: dict[str, int]) -> torch.Tensor:
    """
    Build an embedding matrix aligned to vocabulary (NEEDS {word: index} dictionary)

    Args:
        vocab: {word: index} dict

    Returns:
        matrix: Tensor of shape (len(vocab), 768)
    """
    static_embeddings = safety_bert.embeddings.word_embeddings.weight.detach()
    # shape: (30522, 768) — one vector per BERT subword token

    matrix = torch.zeros(len(vocab), 768)
    found, oov = 0, []

    for word, idx in vocab.items():
        # BERT uses WordPiece — a word may split into multiple subword tokens
        subword_ids = tokenizer.encode(word, add_special_tokens=False)
        if subword_ids:
            # Average subword embeddings to get one vector per word
            matrix[idx] = static_embeddings[subword_ids].mean(dim=0)
            found += 1
        else:
            oov.append(word)

    print(f"Coverage: {found}/{len(vocab)} words "
          f"({100*found/len(vocab):.1f}%)")
    if oov:
        print(f"OOV sample: {oov[:10]}")
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: Contextual sentence embeddings  (better input signal for Bi-GRU)
# Run each incident report through safetyBERT and get per-token hidden states.
# Your Bi-GRU then processes these instead of a simple embedding lookup.
# ─────────────────────────────────────────────────────────────────────────────
def get_contextual_embeddings(
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu"
) -> list[torch.Tensor]:
    """
    Encode a list of raw incident descriptions into contextual token embeddings.

    Args:
        texts:      List of raw description strings (before your preprocessing)
        batch_size: How many documents to encode at once
        max_length: BERT max tokens — 128 is fine for incident reports
        device:     'cuda' or 'cpu'

    Returns:
        List of tensors, each shape (seq_len, 768) — one per document.
        seq_len varies per document (padding removed).
    """
    safety_bert.to(device)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = safety_bert(**inputs)

        # last_hidden_state: (batch, seq_len, 768)
        hidden = outputs.last_hidden_state

        # Strip padding — return only real tokens per document
        for j, length in enumerate(inputs["attention_mask"].sum(dim=1)):
            # slice off [CLS] and [SEP] too — your Bi-GRU doesn't need them
            all_embeddings.append(hidden[j, 1:length-1, :].cpu())

    return all_embeddings


# %%
vocab = {
    "worker": 0,
    "fell": 1,
    "on": 2,
    "ladder": 3
}

flagged_df = pd.concat(flagged_rows).drop_duplicates(subset='reference')
flagged_df.to_csv("labelling_review_flagged.csv", index=False)
print(f"Exported {len(flagged_df)} flagged cases to labelling_review_flagged.csv")

# %% [markdown]
# ## New Section

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

matrix = get_embedding_matrix(vocab)
print(matrix.shape)  # → torch.Size([4, 768])
print(matrix)
