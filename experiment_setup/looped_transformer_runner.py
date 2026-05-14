"""Experiment runner utilities for the LoopedTransformer classifier."""

import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer as _AutoTokenizer

from implementations.looped_transformer import LoopedTransformer
from modules.data_loader.bert_loader import df_to_bert_dataloader
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_tokenizer import BertTokenizerWrapper
from modules.encoding import LabelEncoder
from modules.training_loop import training

_LOOPED_DEFAULTS = {
    "model_name": "adanish91/safetybert",
    "d_model": 256,
    "nhead": 8,
    "dim_feedforward": 1024,
    "num_loops": 6,
    "dropout": 0.1,
    "max_seq_len": 512,
    "max_length": None,          # None → auto-computed from 95th percentile of train lengths
    "max_length_percentile": 95,
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "criterion_type": "focal",
    "best_metric": "f1_macro",
    "scheduler_step_per_batch": False,
    "run_name": None,
    "save": True,
    "log_leaderboard": True,
    "verbose": True,
}


def _resolve_max_length(train_df, text_col, model_name, percentile):
    """Compute max_length as the given percentile of tokenised training lengths."""
    tok = _AutoTokenizer.from_pretrained(model_name)
    lengths = [len(tok(t, truncation=False)["input_ids"]) for t in train_df[text_col]]
    return int(np.percentile(lengths, percentile))


def looped_transformer_encode(train_df, valid_df, test_df, text_col, energy_model, cfg):
    """Tokenise splits and encode labels for the LoopedTransformer.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the raw-text column.
        energy_model: If ``True``, use ``energy_type``; otherwise ``potential_damage``.
        cfg: Resolved config dict (see :func:`looped_transformer_train`).

    Returns:
        tuple: ``(train_dl, valid_dl, test_dl, label_enc, vocab_size, max_length)``
    """
    label_col = "energy_type" if energy_model else "potential_damage"

    label_enc = LabelEncoder()
    label_enc.fit(train_df[label_col].tolist())

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df  = test_df.copy()
    for df in (train_df, valid_df, test_df):
        df["_label"] = label_enc.encode_many(df[label_col].tolist())

    model_name = cfg["model_name"]
    max_length = cfg["max_length"] or _resolve_max_length(
        train_df, text_col, model_name, cfg["max_length_percentile"]
    )

    tokenizer = BertTokenizerWrapper(
        BertEmbeddingConfig(model_name=model_name, max_length=max_length)
    )
    batch_size = cfg["batch_size"]

    train_dl = df_to_bert_dataloader(train_df, text_col, "_label", tokenizer, batch_size=batch_size, shuffle=True)
    valid_dl = df_to_bert_dataloader(valid_df, text_col, "_label", tokenizer, batch_size=batch_size * 2, shuffle=False)
    test_dl  = df_to_bert_dataloader(test_df,  text_col, "_label", tokenizer, batch_size=batch_size * 2, shuffle=False)

    tok = _AutoTokenizer.from_pretrained(model_name)
    vocab_size = tok.vocab_size

    return train_dl, valid_dl, test_dl, label_enc, vocab_size, max_length


def looped_transformer_train(
    train_dl, valid_dl, test_dl,
    label_enc, vocab_size,
    energy_model=True,
    train_config=None,
):
    """Build and train a LoopedTransformer.

    Accepts the tuple produced by :func:`looped_transformer_encode` (unpack
    with positional args after ``*encoded``).

    Args:
        train_dl: Training DataLoader.
        valid_dl: Validation DataLoader.
        test_dl: Test DataLoader.
        label_enc: Fitted :class:`~modules.encoding.LabelEncoder`.
        vocab_size: Tokeniser vocabulary size (used to size the embedding table).
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        train_config: Optional dict of overrides. Supported keys:

            - ``model_name`` (str, default ``"adanish91/safetybert"``) — HuggingFace tokenizer
            - ``d_model`` (int, default ``256``)
            - ``nhead`` (int, default ``8``)
            - ``dim_feedforward`` (int, default ``1024``)
            - ``num_loops`` (int, default ``6``)
            - ``dropout`` (float, default ``0.1``)
            - ``max_seq_len`` (int, default ``512``)
            - ``batch_size`` (int, default ``32``)
            - ``epochs`` (int, default ``50``)
            - ``patience`` (int, default ``10``)
            - ``criterion_type`` (str, default ``"focal"``)
            - ``best_metric`` (str, default ``"f1_macro"``)
            - ``run_name`` (str | None)
            - ``save`` (bool, default ``True``)
            - ``log_leaderboard`` (bool, default ``True``)
            - ``verbose`` (bool, default ``True``)
            - ``optimizer_fn``: ``Callable[[nn.Module], Optimizer]``
            - ``scheduler_fn``: ``Callable[[Optimizer], LRScheduler]``

    Returns:
        dict: The result dictionary returned by
        :func:`~modules.training_loop.training`.
    """
    cfg = {**_LOOPED_DEFAULTS, **(train_config or {})}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LoopedTransformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        num_loops=cfg["num_loops"],
        num_classes=label_enc.num_classes,
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer_fn = cfg.pop("optimizer_fn", None)
    scheduler_fn = cfg.pop("scheduler_fn", None)
    optimiser = optimizer_fn(model) if optimizer_fn is not None else None
    scheduler = scheduler_fn(optimiser) if (scheduler_fn is not None and optimiser is not None) else None

    return training(
        model=model,
        model_type="looped_transformer",
        energy_model=energy_model,
        need_length=False,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        num_classes=label_enc.num_classes,
        class_dict=label_enc.id_to_label,
        optimiser=optimiser,
        scheduler=scheduler,
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        criterion_type=cfg["criterion_type"],
        best_metric=cfg["best_metric"],
        scheduler_step_per_batch=cfg["scheduler_step_per_batch"],
        run_name=cfg["run_name"],
        save=cfg["save"],
        log_leaderboard=cfg["log_leaderboard"],
        verbose=cfg["verbose"],
        device=device,
    )


def looped_transformer_run_single(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    train_config=None,
):
    """Run one end-to-end LoopedTransformer experiment on a single split.

    Tokenises the splits, encodes labels, then trains one model.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the raw-text column.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        train_config: Optional training configuration overrides (see
            :func:`looped_transformer_train`).

    Returns:
        dict: The result dictionary returned by
        :func:`~modules.training_loop.training`.
    """
    cfg = {**_LOOPED_DEFAULTS, **(train_config or {})}
    train_dl, valid_dl, test_dl, label_enc, vocab_size, max_length = looped_transformer_encode(
        train_df, valid_df, test_df, text_col, energy_model, cfg
    )
    result = looped_transformer_train(
        train_dl, valid_dl, test_dl, label_enc, vocab_size,
        energy_model=energy_model,
        train_config=train_config,
    )
    if (
        result.get("best_model_state_dict") is not None
        and result["config"].get("save")
    ):
        merged = cfg
        save_dir = Path(result["config"]["save_dir"])
        save_name = result["config"]["save_name"]
        artifacts = {
            "label_enc": label_enc,
            "max_length": max_length,
            "model_name": merged["model_name"],
            "energy_model": energy_model,
            "batch_size": merged["batch_size"],
            "vocab_size": vocab_size,
            "d_model": merged["d_model"],
            "nhead": merged["nhead"],
            "dim_feedforward": merged["dim_feedforward"],
            "num_loops": merged["num_loops"],
            "dropout": merged["dropout"],
            "max_seq_len": merged["max_seq_len"],
            "text_col": text_col,
        }
        with open(save_dir / f"{save_name}_artifacts.pkl", "wb") as af:
            pickle.dump(artifacts, af)

    return result


def looped_transformer_run_multiple(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    n=5,
    train_config=None,
):
    """Run multiple LoopedTransformer training runs on the same split.

    Tokenisation and label encoding are done once; :func:`looped_transformer_train`
    is called ``n`` times with fresh model weights each time.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the raw-text column.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        n: Number of training runs.
        train_config: Optional training configuration overrides (see
            :func:`looped_transformer_train`).

    Returns:
        list[dict]: A list of result dictionaries from each run.
    """
    cfg = {**_LOOPED_DEFAULTS, **(train_config or {})}
    train_dl, valid_dl, test_dl, label_enc, vocab_size, _ = looped_transformer_encode(
        train_df, valid_df, test_df, text_col, energy_model, cfg
    )
    return [
        looped_transformer_train(
            train_dl, valid_dl, test_dl, label_enc, vocab_size,
            energy_model=energy_model,
            train_config=train_config,
        )
        for _ in range(n)
    ]


def looped_transformer_hparam_search(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    n_trials=30,
    timeout=None,
):
    """Run Optuna hyperparameter search over LoopedTransformer architecture and training params.

    Tokenisation and label encoding are done once before the study starts.
    Searches over ``d_model``, ``nhead``, ``dim_feedforward``, ``num_loops``,
    ``dropout``, ``lr``, and ``epochs``.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the raw-text column.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        n_trials: Maximum number of Optuna trials.
        timeout: Stop after this many seconds regardless of ``n_trials``
            (``None`` = no limit).

    Returns:
        optuna.Study: Inspect with ``study.best_params``, ``study.best_value``,
        and ``optuna.visualization`` helpers.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_cfg = {**_LOOPED_DEFAULTS}
    train_dl, valid_dl, test_dl, label_enc, vocab_size, _ = looped_transformer_encode(
        train_df, valid_df, test_df, text_col, energy_model, base_cfg
    )

    def objective(trial):
        d_model       = trial.suggest_categorical("d_model", [128, 256, 512])
        nhead         = trial.suggest_categorical("nhead", [4, 8])
        dim_ff_mult   = trial.suggest_categorical("dim_feedforward_mult", [2, 4])
        num_loops     = trial.suggest_int("num_loops", 2, 12)
        dropout       = trial.suggest_float("dropout", 0.05, 0.4)
        lr            = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        epochs        = trial.suggest_int("epochs", 20, 80, step=10)

        # nhead must divide d_model — skip invalid combos
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        def optimizer_fn(model):
            return torch.optim.Adam(model.parameters(), lr=lr)

        cfg = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": d_model * dim_ff_mult,
            "num_loops": num_loops,
            "dropout": dropout,
            "epochs": epochs,
            "optimizer_fn": optimizer_fn,
            "run_name": f"looped_hparam_trial_{trial.number}",
            "save": True,
            "log_leaderboard": True,
            "verbose": False,
        }
        result = looped_transformer_train(
            train_dl, valid_dl, test_dl, label_enc, vocab_size,
            energy_model=energy_model,
            train_config=cfg,
        )
        return result["best_metric_value"]

    study = optuna.create_study(direction="maximize", study_name="looped_transformer_hparam_search")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study
