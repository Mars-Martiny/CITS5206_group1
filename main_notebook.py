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

# %%
import pandas as pd

df = pd.read_csv("dataset/final_dataset.csv")
print(df.columns)

# %%
## Need to install contractions. Need to list this!
# %pip install contractions

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

# oneTextPreProcessor = OneTextPreProcessor(keep_numbers=True, column_map=column_map, lemmatize=True, lemma_config=lemma_config)
# mod_df = oneTextPreProcessor.pre_process_df(df, column_map["Detailed Description of Event"])
# mod_df

# %%
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

# %%
tokens_col = f"description_tokens_lemma"

# %%
from implementations.tf_idf import TFIDFClassifier, TFIDFVectorizer, build_tfidf_dataloader
from modules.training_loop import _build_train_config, training

# %%
import torch
from modules.encoding import LabelEncoder

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

model.eval()
all_preds, val_labels = [], []

with torch.no_grad():
    for batch in valid_dl:
        D, _, Energy, Risk = batch
        D = D.to(device)
        logits = model(D)
        all_preds.append(logits.argmax(dim=1).cpu())
        val_labels.append(Energy.cpu())  # or Risk if energy_model=False

preds = torch.cat(all_preds).numpy()
val_labels = torch.cat(val_labels).numpy()

labels = list(range(label_enc.num_classes))
target_names = label_enc.decode_many(labels)

print(classification_report(val_labels, preds, labels=labels, target_names=target_names))

# %%
from sklearn.metrics import f1_score
import numpy as np

# find labels that actually appear in val set
present = np.unique(val_labels)
print(f1_score(val_labels, preds, labels=present, average='macro'))

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(val_labels, preds, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(14,12))
disp.plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# %%
import pandas as pd

true_class = "Susceptible Part"
pred_class = "Specialized Shape"

true_idx = label_enc.encode(true_class)
pred_idx = label_enc.encode(pred_class)

mask = (val_labels == true_idx) & (preds == pred_idx)
misclassified_indices = mask.nonzero()[0]

text_col = column_map["Detailed Description of Event"]
misclassified_texts = model1_valid.iloc[misclassified_indices][[text_col, "energy_type"]].copy()
misclassified_texts["predicted"] = pred_class

print(f"Found {len(misclassified_indices)} sample(s) labelled '{true_class}' but predicted as '{pred_class}':\n")
pd.set_option("display.max_colwidth", None)
misclassified_texts.reset_index(drop=True)


# %%
model1_valid.columns

# %%
# Look at Specialized Shape examples in validation
spec_shape = model1_valid[model1_valid['energy_type'] == 'Specialized Shape']

# Look at Susceptible Part examples in validation
susc_part = model1_valid[model1_valid['energy_type'] == 'Susceptible Part']

# %%
compare = pd.concat([spec_shape, susc_part])[['reference', 'energy_type', 'description', 'description_tokens']]
compare.to_csv("debugging.csv")

# %%
import pandas as pd
from modules.inference import run_inference

# Run inference on the validation set
infer_config = {
    "model": model,
    "device": device,
    "need_length": False,
    "energy_model": True,
    "test_dl": val_dl,
}
infer_result = run_inference(infer_config)

all_preds = infer_result["all_preds"].numpy()
all_true  = infer_result["all_targets"].numpy()

# Decode labels
pred_labels = label_enc.decode_many(all_preds)
true_labels = label_enc.decode_many(all_true)

# Build results df — valid_dl was built without shuffling, so order is preserved
results_df = model1_valid.copy().reset_index(drop=True)
results_df['pred_label'] = pred_labels
results_df['true_label'] = true_labels
results_df['correct'] = results_df['pred_label'] == results_df['true_label']

# ── Priority review pairs ──────────────────────────────────────────────────
review_pairs = [
    ("Specialized Shape", "Susceptible Part"),
    ("Susceptible Part",  "Specialized Shape"),
    ("Machine",           "Vehicular"),
    ("Vehicular",         "Machine"),
    ("Gravitational",     "Human"),
    ("Human",             "Gravitational"),
    ("Object",            "Gravitational"),
    ("Gravitational",     "Object"),
    ("Thermal",           "Object"),
    ("Thermal",           "Other"),
]

cols = ['reference', 'true_label', 'pred_label', 'description', 'description_tokens']

for true_cls, pred_cls in review_pairs:
    mask = (
        (results_df['true_label'] == true_cls) &
        (results_df['pred_label'] == pred_cls)
    )
    subset = results_df[mask][cols]
    if subset.empty:
        continue

valid_df = model1_valid.copy().reset_index(drop=True)
valid_df["pred_label"] = pred_labels
valid_df["true_label"] = true_labels
valid_df["correct"]    = valid_df["pred_label"] == valid_df["true_label"]

print(f"Test accuracy : {valid_df['correct'].mean():.3f}  ({valid_df['correct'].sum()}/{len(valid_df)})")
valid_df.head(3)



# %%
# ── Inference on test set ─────────────────────────────────────────────────
test_tokenized_docs = model1_test[tokens_col].tolist()
test_labels_enc     = torch.tensor(label_enc.encode_many(model1_test[label_col].tolist()))
test_vecs           = vectorizer.transform(test_tokenized_docs)
test_dl             = build_tfidf_dataloader(test_vecs, test_labels_enc, shuffle=False)

test_infer_config = {
    "model": model,
    "device": device,
    "need_length": False,
    "energy_model": True,
    "test_dl": test_dl,
}
test_result = run_inference(test_infer_config)

test_pred_labels = label_enc.decode_many(test_result["all_preds"].numpy())
test_true_labels = label_enc.decode_many(test_result["all_targets"].numpy())

test_df = model1_test.copy().reset_index(drop=True)
test_df["pred_label"] = test_pred_labels
test_df["true_label"] = test_true_labels
test_df["correct"]    = test_df["pred_label"] == test_df["true_label"]

print(f"Test accuracy : {test_df['correct'].mean():.3f}  ({test_df['correct'].sum()}/{len(test_df)})")
test_df.head(3)


# %%
# Export all misclassified priority pairs to CSV for client review
flagged_rows = []
for true_cls, pred_cls in review_pairs:
    mask = (
        (results_df['true_label'] == true_cls) &
        (results_df['pred_label'] == pred_cls)
    )
    flagged_rows.append(results_df[mask][cols])

flagged_df = pd.concat(flagged_rows).drop_duplicates(subset='reference')
flagged_df.to_csv("labelling_review_flagged.csv", index=False)
print(f"Exported {len(flagged_df)} flagged cases to labelling_review_flagged.csv")
