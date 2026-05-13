# TF‑IDF Results (Leaderboard Summary)

This report summarizes TF‑IDF performance using runs recorded in `leaderboard/leaderboard.csv`, and highlights the top-performing runs for each TF‑IDF task variant.

## Overall performance (all TF‑IDF runs)

- **Number of runs**: 90
- **Validation macro‑F1 (primary metric)**:
  - **Mean**: 0.5825 (std: 0.0673)
  - **Min / Median / Max**: 0.2290 / 0.6015 / 0.6420
  - **25% / 75% quantiles**: 0.5763 / 0.6136
- **Validation accuracy**:
  - **Mean**: 0.6983
  - **Max**: 0.7875
- **Validation loss**:
  - **Best (min)**: 0.5098
- **Training time**:
  - **Mean**: 1.075 s
  - **Max**: 4.596 s

### Benchmark takeaway

Across all TF‑IDF runs, the **best validation macro‑F1 is ~63%** (max: 0.6420; upper quartile: 0.6136). We treat **~63% macro‑F1** as a **baseline benchmark for “basic functionality”** of a TF‑IDF model on this project.

## Top performance (per task)

### Best for potential damage (`energy_model=False`)

- **Run name**: `tf_idf_run_20260427_174823`
- **Val macro‑F1**: **0.6420**
- **Val accuracy**: 0.7875
- **Val weighted‑F1**: 0.7811
- **Val loss**: 0.5098
- **Training details**: `epochs_max=60`, `patience=15`, `best_epoch=2`, `lr=0.009772`, scheduler=`StepLR`
- **Saved model**: `trained_models/20260427_174823_tf_idf/tf_idf_model.pt`

### Best for energy type (`energy_model=True`)

- **Run name**: `tf_idf_run_20260427_174505`
- **Val macro‑F1**: **0.6371**
- **Val accuracy**: 0.6667
- **Val weighted‑F1**: 0.6596
- **Val loss**: 1.1210
- **Training details**: `epochs_max=80`, `patience=15`, `best_epoch=6`, `lr=0.001540`, scheduler=`StepLR`
- **Saved model**: `trained_models/20260427_174505_tf_idf/tf_idf_model.pt`

## Best parameters (from notebook output)

The notebook code (`main_notebook.py`) performs an Optuna search and prints:
`print("Best params       :", hparam_study.best_params)`.

However, in the current saved `main_notebook.ipynb` content, the printed Optuna *output values* (the actual dict of best params) are not present/searchable, so the exact Optuna `best_params` dict cannot be recovered from the notebook artifact alone.

What we *can* report from the **best runs above** (these are logged per-run in `leaderboard/leaderboard.csv`):

- **lr**: 0.0097721189210879
- **epochs_max**: 60
- **patience**: 15
- **scheduler**: `StepLR` (stepped per epoch; `scheduler_step_per_batch=False`)
- **energy_model**: False

Note: Optuna also searches over `hidden_dim` (and other items), but `hidden_dim` is not currently recorded in `leaderboard/leaderboard.csv`. If needed, we should either (a) re-run the Optuna cell to re-print `best_params`, or (b) log `hidden_dim` into the leaderboard for each TF‑IDF run.

# Fundamental Limitation: Bag-of-Words Has No Word Order

TF-IDF is a **bag-of-words** model. It converts a document into a vector of token frequencies weighted by inverse document frequency, but the order of those tokens is completely discarded. This is a structural limitation that cannot be fixed by tuning hyperparameters.

## The consequence for incident classification

Consider two incident descriptions:

> _"Worker fell on the truck."_

> _"Truck fell on the worker."_

Both sentences contain exactly the same words. TF-IDF produces **identical feature vectors** for both, so the classifier assigns them the same prediction — even though the first is likely a fall from height and the second is a struck-by event with entirely different energy type and damage implications.

This is not an edge case. Incident reports commonly describe the same actors and objects in different causal roles: a person can fall _onto_ a surface or a surface can fall _onto_ a person; equipment can strike a worker or a worker can strike equipment. The label depends on who is acting and who is receiving the action, which is encoded in word order, not word presence.

## What sequence models gain

Bi-GRU and transformer-based models process tokens in order and build hidden states that capture subject-verb-object relationships. The same two sentences above would produce different hidden-state sequences and, in a well-trained model, different predictions. This is the primary structural reason those architectures can outperform TF-IDF on this task even when TF-IDF has more hyperparameter tuning.

TF-IDF is still useful as a fast, interpretable baseline — its macro-F1 of ~0.63 represents the ceiling of what word-frequency signal alone can achieve on this dataset.

---

# Negative Results
## Using Transform Average Weighting and L2 Normalization did not work
By using doc_vec = sum(tfidf_score(word) * embed(word) for word in doc) / sum(tfidf_score(word) for word in doc)
it results in all entries have the same embedding, possibly because IDF collapses document-level variance.
Following are captured:
```
len(vectorizer.vocab) 2156
E.shape torch.Size([2156, 768])
E.sum tensor(-36086.6562)
E.std tensor(0.0360)
E.train_vecs[:5] tensor([[-0.0219, -0.0628, -0.0251,  ..., -0.0168, -0.0525, -0.0081],
        [-0.0219, -0.0628, -0.0251,  ..., -0.0168, -0.0525, -0.0081],
        [-0.0219, -0.0628, -0.0251,  ..., -0.0168, -0.0525, -0.0081],
        [-0.0219, -0.0628, -0.0251,  ..., -0.0168, -0.0525, -0.0081],
        [-0.0219, -0.0628, -0.0251,  ..., -0.0168, -0.0525, -0.0081]]) <== All have same embedding
Train vecs dim tensor(2.8667e-09) <== Very close to 10
```

Same issue happened when doing L2 Norm

## Using Raw Value from static embedding also did not work
While the value is no longer all the same
```
len(vectorizer.vocab) 2156
E.shape torch.Size([2156, 768])
E.sum tensor(-36086.6562)
E.std tensor(0.0360)
E.train_vecs[:5] tensor([[-0.0920, -0.2636, -0.1055,  ..., -0.0705, -0.2204, -0.0341],
        [-0.0917, -0.2629, -0.1052,  ..., -0.0703, -0.2198, -0.0340],
        [-0.0898, -0.2574, -0.1030,  ..., -0.0689, -0.2152, -0.0333],
        [-0.0965, -0.2764, -0.1106,  ..., -0.0739, -0.2311, -0.0358],
        [-0.0718, -0.2058, -0.0824,  ..., -0.0551, -0.1721, -0.0266]])
Train vecs dim tensor(0.0135)
```

The result is just very bad, with `Best f1_macro: 0.034906` and the model only predicts 2 classes out of all the classes.

Consulting with Claude gave this result:
```
It's the architecture mismatch. You're trying to use a TFIDFClassifier (a simple 2-layer feedforward net) on static averaged BERT embeddings — and that combination has a core tension:

Static embeddings from SafetyBERT are contextual by design — they're meant to be used with a sequence model like your Bi-GRU, not averaged into a single vector and fed into a linear layer
Averaging 768-dim vectors across all tokens in a document inherently destroys the positional and contextual information that makes BERT embeddings useful
The TFIDFClassifier is simply too shallow to recover signal from such a compressed representation
```