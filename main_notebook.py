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

# %% [markdown]
# # Quick TF-IDF + SVM Run
# This notebook only runs TF-IDF + SVM and logs the result to leaderboard.
# Other expensive model searches are intentionally disabled.

# %%
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# %%
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

with open("column_map.json", "r") as file:
    column_map = json.load(file)

text_col = column_map["Detailed Description of Event"]

# %%
energy_train = pd.read_csv("dataset/model1_train.csv").rename(columns=column_map)
energy_valid = pd.read_csv("dataset/model1_valid.csv").rename(columns=column_map)
energy_test = pd.read_csv("dataset/model1_test.csv").rename(columns=column_map)

damage_train = pd.read_csv("dataset/model2_train.csv").rename(columns=column_map)
damage_valid = pd.read_csv("dataset/model2_valid.csv").rename(columns=column_map)
damage_test = pd.read_csv("dataset/model2_test.csv").rename(columns=column_map)

# %% [markdown]
# ## Helper functions

# %%
def _guess_label_column(df, energy_model=True):
    """Return correct label column."""
    return "energy_type" if energy_model else "potential_damage"

def _build_metrics(y_true, y_pred):
    """Build metrics compatible with leaderboard-style reporting."""
    labels = sorted(set(y_true) | set(y_pred))

    return {
        "loss": None,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "class_metrics": {},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "auto_classification_rate": None,
        "fatal_flag_count": None,
        "fatal_flag_rate": None,
        "meets_requirement": None,
        "threshold_used": None,
        "confidence_high_rate": None,
        "confidence_medium_rate": None,
        "confidence_low_rate": None,
        "req_high_confidence_met": None,
        "fatal_accuracy": None,
        "req_fatal_accuracy_met": None,
        "per_class_requirements": None,
        "req_all_f1_targets_met": None,
    }


def run_tfidf_svm(
    train_df,
    valid_df,
    test_df,
    text_col,
    energy_model=True,
    run_name=None,
    leaderboard_dir="leaderboard",
    log_leaderboard=True,
):
    """Train TF-IDF + SVM, evaluate on validation/test sets, and log to leaderboard."""
    start_time = time.time()

    label_col = _guess_label_column(train_df, energy_model=energy_model)

    task_name = "energy" if energy_model else "damage"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"tfidf_svm_{task_name}_{timestamp}"

    X_train = train_df[text_col].fillna("").astype(str)
    y_train = train_df[label_col]

    X_valid = valid_df[text_col].fillna("").astype(str)
    y_valid = valid_df[label_col]

    X_test = test_df[text_col].fillna("").astype(str)
    y_test = test_df[label_col]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=20000,
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    valid_metrics = _build_metrics(y_valid.tolist(), valid_pred.tolist())
    test_metrics = _build_metrics(y_test.tolist(), test_pred.tolist())

    print(f"\n===== {run_name} =====")
    print(f"Validation accuracy: {valid_metrics['accuracy']:.4f}")
    print(f"Validation F1 macro: {valid_metrics['f1_macro']:.4f}")
    print(f"Test accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"Test F1 macro:       {test_metrics['f1_macro']:.4f}")

    run_summary = {
    "history": {
        "training": {
            "val": {
                key: [value]
                for key, value in valid_metrics.items()
            }
        },
        "test": {
            key: [value]
            for key, value in test_metrics.items()
        },
    },
    "metrics": test_metrics,
    "test_metrics": test_metrics,
    "best_epoch": 1,
    "best_metric_name": "test_f1_macro",
    "best_metric_value": test_metrics["f1_macro"],
    "best_model_state_dict": None,
    "training_time_sec": time.time() - start_time,
}

    config = {
        "best_metric": "test_f1_macro",
        "run_name": run_name,
        "save_name": run_name,
        "model_type": "TFIDF_SVM",
        "model": model,
        "energy_model": energy_model,
        "epochs": 1,
        "patience": None,
        "optimiser": None,
        "scheduler": None,
        "criterion_type": None,
        "scheduler_step_per_batch": None,
        "clip_grad_max_norm": None,
        "class_dict": {},
        "num_classes": len(set(y_train.tolist()) | set(y_valid.tolist()) | set(y_test.tolist())),
        "metadata": {
            "model_class": "LinearSVC",
            "feature_representation": "TF-IDF",
            "classifier_class": "LinearSVC",
            "optimiser_class": None,
            "scheduler_class": None,
            "criterion_class": None,
        }
    }

    if log_leaderboard:
        from modules.leaderboard.logger import log_run

        log_run(
            run_summary=run_summary,
            config=config,
            model_path=None,
            leaderboard_dir=leaderboard_dir,
        )

    return {
        "model": model,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "valid_predictions": valid_pred,
        "test_predictions": test_pred,
    }

# %% [markdown]
# ## Run TF-IDF + SVM for Energy

# %%
tfidf_svm_energy_result = run_tfidf_svm(
    train_df=energy_train,
    valid_df=energy_valid,
    test_df=energy_test,
    text_col=text_col,
    energy_model=True,
    run_name="tfidf_svm_energy_quick",
    log_leaderboard=True,
)

# %% [markdown]
# ## Run TF-IDF + SVM for Damage

# %%
tfidf_svm_damage_result = run_tfidf_svm(
    train_df=damage_train,
    valid_df=damage_valid,
    test_df=damage_test,
    text_col=text_col,
    energy_model=False,
    run_name="tfidf_svm_damage_quick",
    log_leaderboard=True,
)