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
# # General Common Config

# %%
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

import json
import pandas as pd

with open('column_map.json', 'r') as file:
    column_map = json.load(file)

text_col = column_map["Detailed Description of Event"]

# %%
energy_train = pd.read_csv("dataset/model1_train.csv").rename(columns=column_map)
energy_valid = pd.read_csv("dataset/model1_valid.csv").rename(columns=column_map)
energy_test  = pd.read_csv("dataset/model1_test.csv").rename(columns=column_map)

damage_train = pd.read_csv("dataset/model2_train.csv").rename(columns=column_map)
damage_valid = pd.read_csv("dataset/model2_valid.csv").rename(columns=column_map)
damage_test  = pd.read_csv("dataset/model2_test.csv").rename(columns=column_map)

# %% [markdown]
# # TF-IDF Vanilla (`tfidf`)

# %%
from experiment_setup.tf_idf_runner import tf_idf_hparam_search

study_tfidf_energy = tf_idf_hparam_search(
    energy_train, energy_valid, energy_test, text_col,
    lemma_config=lemma_config, energy_model=True, n_trials=40,
)

# %%
study_tfidf_damage = tf_idf_hparam_search(
    damage_train, damage_valid, damage_test, text_col,
    lemma_config=lemma_config, energy_model=False, n_trials=40,
)

# %% [markdown]
# # TF-IDF Safe Static (`tfidf_safe_static`)
# TF-IDF weighted average of SafetyBERT static token embeddings.

# %%
import torch
import optuna
from experiment_setup.tf_idf_runner import (
    pre_process_dataset as _tfidf_pp,
    tf_idf_encode,
    tf_idf_train,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_tfidf_ss_objective(encoded, run_prefix):
    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        epochs     = trial.suggest_int("epochs", 30, 150, step=10)
        sched      = trial.suggest_categorical("scheduler", ["cosine", "cosine_warmup", "step", "none"])

        def optimizer_fn(m):
            return torch.optim.Adam(m.parameters(), lr=lr)

        def scheduler_fn(opt):
            if sched == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
            if sched == "cosine_warmup":
                w = max(1, epochs // 10)
                return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                    torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, w),
                    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - w, eta_min=1e-6),
                ], milestones=[w])
            if sched == "step":
                return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
            return False

        cfg = {
            "epochs": epochs, "hidden_dim": hidden_dim,
            "optimizer_fn": optimizer_fn, "scheduler_fn": scheduler_fn,
            "scheduler_step_per_batch": False, "patience": 15,
            "best_metric": "f1_macro", "save": True, "log_leaderboard": True, "verbose": False,
            "feature_representation": "tfidf_embed_avg",
            "embedding_model_name": "adanish91/safetybert",
            "run_name": f"{run_prefix}_trial_{trial.number}",
        }
        artifact_extras = {"text_col": text_col, "lemma_config": lemma_config, "keep_numbers": False}
        return tf_idf_train(*encoded, train_config=cfg, artifact_extras=artifact_extras)["best_metric_value"]
    return objective


_tfidf_ss_enc_e = tf_idf_encode(
    *_tfidf_pp(energy_train, energy_valid, energy_test, text_col, False, lemma_config),
    text_col, lemma_config, True,
)
study_tfidf_ss_energy = optuna.create_study(direction="maximize", study_name="tfidf_safe_static_energy")
study_tfidf_ss_energy.optimize(
    _make_tfidf_ss_objective(_tfidf_ss_enc_e, "tfidf_safe_static"),
    n_trials=40, show_progress_bar=True,
)

# %%
_tfidf_ss_enc_d = tf_idf_encode(
    *_tfidf_pp(damage_train, damage_valid, damage_test, text_col, False, lemma_config),
    text_col, lemma_config, False,
)
study_tfidf_ss_damage = optuna.create_study(direction="maximize", study_name="tfidf_safe_static_damage")
study_tfidf_ss_damage.optimize(
    _make_tfidf_ss_objective(_tfidf_ss_enc_d, "tfidf_safe_static"),
    n_trials=40, show_progress_bar=True,
)

# %% [markdown]
# # BiGRU Vanilla (`bi_gru`)

# %%
from experiment_setup.bi_gru_runner import bigru_hparam_search

study_bigru_energy = bigru_hparam_search(
    energy_train, energy_valid, energy_test, text_col,
    lemma_config=lemma_config, energy_model=True, n_trials=40,
)

# %%
study_bigru_damage = bigru_hparam_search(
    damage_train, damage_valid, damage_test, text_col,
    lemma_config=lemma_config, energy_model=False, n_trials=40,
)

# %% [markdown]
# # BiGRU Safe Static (`bi_gru_safe_static`)
# Token sequences encoded once; SafetyBERT embedding matrix injected into the embedding layer per trial.

# %%
from experiment_setup.bi_gru_runner import (
    pre_process_dataset as _bigru_pp,
    bigru_encode,
    bigru_train,
)


def _make_bigru_static_objective(encoded, energy_model, run_prefix):
    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        epochs     = trial.suggest_int("epochs", 20, 100, step=10)
        sched      = trial.suggest_categorical("scheduler", ["cosine", "cosine_warmup", "step", "none"])

        def optimizer_fn(m):
            return torch.optim.Adam(m.parameters(), lr=lr)

        def scheduler_fn(opt):
            if sched == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
            if sched == "cosine_warmup":
                w = max(1, epochs // 10)
                return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                    torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, w),
                    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - w, eta_min=1e-6),
                ], milestones=[w])
            if sched == "step":
                return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
            return False

        cfg = {
            "epochs": epochs, "hidden_dim": hidden_dim,
            "embedding_type": "static",
            "embedding_model_name": "adanish91/safetybert",
            "optimizer_fn": optimizer_fn, "scheduler_fn": scheduler_fn,
            "scheduler_step_per_batch": False, "patience": 12,
            "best_metric": "f1_macro", "save": True, "log_leaderboard": True, "verbose": False,
            "run_name": f"{run_prefix}_trial_{trial.number}",
        }
        artifact_extras = {"text_col": text_col, "lemma_config": lemma_config, "keep_numbers": False}
        return bigru_train(*encoded, energy_model=energy_model, train_config=cfg, artifact_extras=artifact_extras)["best_metric_value"]
    return objective


_bigru_ss_enc_e = bigru_encode(
    *_bigru_pp(energy_train, energy_valid, energy_test, text_col, False, lemma_config),
    text_col, lemma_config,
)
study_bigru_ss_energy = optuna.create_study(direction="maximize", study_name="bi_gru_safe_static_energy")
study_bigru_ss_energy.optimize(
    _make_bigru_static_objective(_bigru_ss_enc_e, True, "bi_gru_safe_static"),
    n_trials=40, show_progress_bar=True,
)

# %%
_bigru_ss_enc_d = bigru_encode(
    *_bigru_pp(damage_train, damage_valid, damage_test, text_col, False, lemma_config),
    text_col, lemma_config,
)
study_bigru_ss_damage = optuna.create_study(direction="maximize", study_name="bi_gru_safe_static_damage")
study_bigru_ss_damage.optimize(
    _make_bigru_static_objective(_bigru_ss_enc_d, False, "bi_gru_safe_static"),
    n_trials=40, show_progress_bar=True,
)

# %% [markdown]
# # BiGRU Safe Context (`bi_gru_safe_context`)
# Contextual BERT embeddings pre-computed once; GRU stack retrained each trial.

# %%
from experiment_setup.bi_gru_runner import bigru_contextual_encode


def _make_bigru_context_objective(encoded, energy_model, run_prefix):
    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        epochs     = trial.suggest_int("epochs", 20, 100, step=10)
        sched      = trial.suggest_categorical("scheduler", ["cosine", "cosine_warmup", "step", "none"])

        def optimizer_fn(m):
            return torch.optim.Adam(m.parameters(), lr=lr)

        def scheduler_fn(opt):
            if sched == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
            if sched == "cosine_warmup":
                w = max(1, epochs // 10)
                return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                    torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, w),
                    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - w, eta_min=1e-6),
                ], milestones=[w])
            if sched == "step":
                return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
            return False

        cfg = {
            "epochs": epochs, "hidden_dim": hidden_dim,
            "embedding_type": "contextual",
            "optimizer_fn": optimizer_fn, "scheduler_fn": scheduler_fn,
            "scheduler_step_per_batch": False, "patience": 12,
            "best_metric": "f1_macro", "save": True, "log_leaderboard": True, "verbose": False,
            "run_name": f"{run_prefix}_trial_{trial.number}",
        }
        artifact_extras = {"text_col": text_col, "lemma_config": lemma_config, "keep_numbers": False}
        return bigru_train(*encoded, energy_model=energy_model, train_config=cfg, artifact_extras=artifact_extras)["best_metric_value"]
    return objective


_bigru_ctx_enc_e = bigru_contextual_encode(
    *_bigru_pp(energy_train, energy_valid, energy_test, text_col, False, lemma_config),
    text_col, embedding_model_name="bert-base-uncased",
)
study_bigru_ctx_energy = optuna.create_study(direction="maximize", study_name="bi_gru_safe_context_energy")
study_bigru_ctx_energy.optimize(
    _make_bigru_context_objective(_bigru_ctx_enc_e, True, "bi_gru_safe_context"),
    n_trials=40, show_progress_bar=True,
)

# %%
_bigru_ctx_enc_d = bigru_contextual_encode(
    *_bigru_pp(damage_train, damage_valid, damage_test, text_col, False, lemma_config),
    text_col, embedding_model_name="bert-base-uncased",
)
study_bigru_ctx_damage = optuna.create_study(direction="maximize", study_name="bi_gru_safe_context_damage")
study_bigru_ctx_damage.optimize(
    _make_bigru_context_objective(_bigru_ctx_enc_d, False, "bi_gru_safe_context"),
    n_trials=40, show_progress_bar=True,
)

# %% [markdown]
# # Looped Transformer Vanilla (`rdt`)

# %%
from experiment_setup.looped_transformer_runner import looped_transformer_hparam_search

study_rdt_energy = looped_transformer_hparam_search(
    energy_train, energy_valid, energy_test, text_col,
    energy_model=True, n_trials=30,
)

# %%
study_rdt_damage = looped_transformer_hparam_search(
    damage_train, damage_valid, damage_test, text_col,
    energy_model=False, n_trials=30,
)

# %% [markdown]
# # Looped Transformer Safe Static (`rdt_safe_static`)
# LoopedTransformer with SafetyBERT static embedding initialisation in the token embedding layer.

# %%
import torch
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from experiment_setup.looped_transformer_runner import (
    _LOOPED_DEFAULTS,
    looped_transformer_encode,
    looped_transformer_train,
)


def _make_rdt_safe_static_objective(train_df, valid_df, test_df, energy_model, run_prefix):
    from modules.embedding.safety_bert_static import get_safety_bert_embedding_matrix

    base_cfg = {**_LOOPED_DEFAULTS, "model_name": "bert-base-uncased"}
    train_dl, valid_dl, test_dl, label_enc, vocab_size, max_length = looped_transformer_encode(
        train_df, valid_df, test_df, text_col, energy_model, base_cfg
    )

    def objective(trial):
        d_model     = trial.suggest_categorical("d_model", [128, 256, 512])
        nhead       = trial.suggest_categorical("nhead", [4, 8])
        dim_ff_mult = trial.suggest_categorical("dim_feedforward_mult", [2, 4])
        num_loops   = trial.suggest_int("num_loops", 2, 12)
        dropout     = trial.suggest_float("dropout", 0.05, 0.4)
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        epochs      = trial.suggest_int("epochs", 20, 80, step=10)
        freeze_emb  = trial.suggest_categorical("freeze_emb", [True, False])

        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        def optimizer_fn(m):
            return torch.optim.Adam(m.parameters(), lr=lr)

        cfg = {
            "d_model": d_model, "nhead": nhead,
            "dim_feedforward": d_model * dim_ff_mult,
            "num_loops": num_loops, "dropout": dropout,
            "epochs": epochs, "optimizer_fn": optimizer_fn,
            "run_name": f"{run_prefix}_trial_{trial.number}",
            "save": True, "log_leaderboard": True, "verbose": False,
        }
        result = looped_transformer_train(
            train_dl, valid_dl, test_dl, label_enc, vocab_size,
            energy_model=energy_model, train_config=cfg,
            text_col=text_col, max_length=max_length,
        )
        return result["best_metric_value"]
    return objective


study_rdt_ss_energy = optuna.create_study(direction="maximize", study_name="rdt_safe_static_energy")
study_rdt_ss_energy.optimize(
    _make_rdt_safe_static_objective(energy_train, energy_valid, energy_test, True, "rdt_safe_static"),
    n_trials=20, show_progress_bar=True,
)

# %%
study_rdt_ss_damage = optuna.create_study(direction="maximize", study_name="rdt_safe_static_damage")
study_rdt_ss_damage.optimize(
    _make_rdt_safe_static_objective(damage_train, damage_valid, damage_test, False, "rdt_safe_static"),
    n_trials=20, show_progress_bar=True,
)

# %% [markdown]
# # Looped Transformer Safe Context (`rdt_safe_context`)
# LoopedTransformer tokenised with `bert-base-uncased`; searches contextual-scale architecture.

# %%
def _make_rdt_safe_context_objective(train_df, valid_df, test_df, energy_model, run_prefix):
    base_cfg = {**_LOOPED_DEFAULTS, "model_name": "bert-base-uncased"}
    train_dl, valid_dl, test_dl, label_enc, vocab_size, max_length = looped_transformer_encode(
        train_df, valid_df, test_df, text_col, energy_model, base_cfg
    )

    def objective(trial):
        d_model     = trial.suggest_categorical("d_model", [128, 256, 512])
        nhead       = trial.suggest_categorical("nhead", [4, 8])
        dim_ff_mult = trial.suggest_categorical("dim_feedforward_mult", [2, 4])
        num_loops   = trial.suggest_int("num_loops", 2, 12)
        dropout     = trial.suggest_float("dropout", 0.05, 0.4)
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        epochs      = trial.suggest_int("epochs", 20, 80, step=10)

        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        def optimizer_fn(m):
            return torch.optim.Adam(m.parameters(), lr=lr)

        cfg = {
            "d_model": d_model, "nhead": nhead,
            "dim_feedforward": d_model * dim_ff_mult,
            "num_loops": num_loops, "dropout": dropout,
            "epochs": epochs, "optimizer_fn": optimizer_fn,
            "run_name": f"{run_prefix}_trial_{trial.number}",
            "save": True, "log_leaderboard": True, "verbose": False,
        }
        return looped_transformer_train(
            train_dl, valid_dl, test_dl, label_enc, vocab_size,
            energy_model=energy_model, train_config=cfg,
            text_col=text_col, max_length=max_length,
        )["best_metric_value"]
    return objective


study_rdt_ctx_energy = optuna.create_study(direction="maximize", study_name="rdt_safe_context_energy")
study_rdt_ctx_energy.optimize(
    _make_rdt_safe_context_objective(energy_train, energy_valid, energy_test, True, "rdt_safe_context"),
    n_trials=20, show_progress_bar=True,
)

# %%
study_rdt_ctx_damage = optuna.create_study(direction="maximize", study_name="rdt_safe_context_damage")
study_rdt_ctx_damage.optimize(
    _make_rdt_safe_context_objective(damage_train, damage_valid, damage_test, False, "rdt_safe_context"),
    n_trials=20, show_progress_bar=True,
)

# %% [markdown]
# # BERT (`bert`)

# %%
from experiment_setup.bert_runner import bert_hparam_search

study_bert_energy = bert_hparam_search(
    energy_train, energy_valid, energy_test, text_col,
    energy_model=True, n_trials=20,
)

# %%
study_bert_damage = bert_hparam_search(
    damage_train, damage_valid, damage_test, text_col,
    energy_model=False, n_trials=20,
)
