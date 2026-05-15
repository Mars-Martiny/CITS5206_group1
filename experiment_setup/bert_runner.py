"""Experiment runner utilities for BERT transformer classification."""

import pickle
from pathlib import Path

from implementations.bert_transformer import run_bert_experiment
from modules.encoding import LabelEncoder

def _label_encoder_from_class_dict(class_dict: dict) -> LabelEncoder:
    """Rebuild a fitted :class:`~modules.encoding.LabelEncoder` from saved ``class_dict``."""
    id_to_label: dict[int, str] = {}
    for k, v in class_dict.items():
        id_to_label[int(k)] = str(v)
    le = LabelEncoder()
    le.id_to_label = id_to_label
    le.label_to_id = {label: idx for idx, label in id_to_label.items()}
    le._fitted = True
    return le


_BERT_DEFAULTS = {
    "run_name": "bert_run",
    "fine_tune": False,
    "pooling": "mean",
    "batch_size": 8,
    "epochs": 5,
    "learning_rate": None,
    "dropout": 0.2,
    "max_length": 160,
    "use_class_weights": False,
    "weight_decay": 0.01,
    "threshold": 0.8,
}


def bert_train(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    train_config=None,
):
    """Train a BERT classifier.

    Thin wrapper around :func:`~implementations.bert_transformer.run_bert_experiment`
    that resolves ``energy_model`` to the appropriate label column and applies
    default hyperparameters.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the column containing raw text.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        train_config: Optional dict of overrides. Supported keys:

            - ``run_name`` (str, default ``"bert_run"``)
            - ``fine_tune`` (bool, default ``False``) — fine-tune BERT weights
            - ``pooling`` (str, default ``"mean"``) — ``"cls"`` or ``"mean"``
            - ``batch_size`` (int, default ``8``)
            - ``epochs`` (int, default ``5``)
            - ``learning_rate`` (float | None) — defaults to ``2e-5`` when
              fine-tuning and ``1e-4`` when frozen
            - ``dropout`` (float, default ``0.2``)
            - ``max_length`` (int, default ``160``)
            - ``use_class_weights`` (bool, default ``False``)
            - ``weight_decay`` (float, default ``0.01``)
            - ``threshold`` (float, default ``0.8``)

    Returns:
        dict: The result dictionary returned by
        :func:`~modules.training_loop.training`.
    """
    cfg = {**_BERT_DEFAULTS, **(train_config or {})}
    label_col = "energy_type" if energy_model else "potential_damage"
    target_type = "energy" if energy_model else "risk"

    result = run_bert_experiment(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        text_col=text_col,
        label_col=label_col,
        target_type=target_type,
        run_name=cfg["run_name"],
        fine_tune=cfg["fine_tune"],
        pooling=cfg["pooling"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        dropout=cfg["dropout"],
        max_length=cfg["max_length"],
        use_class_weights=cfg["use_class_weights"],
        weight_decay=cfg["weight_decay"],
        threshold=cfg["threshold"],
        verbose=cfg.get("verbose", True),
    )

    if (
        result.get("best_model_state_dict") is not None
        and result["config"].get("save")
    ):
        save_dir = Path(result["config"]["save_dir"])
        save_name = result["config"]["save_name"]
        artifacts = {
            "energy_model": energy_model,
            "label_enc": _label_encoder_from_class_dict(dict(result["config"]["class_dict"])),
            "max_length": cfg["max_length"],
            "model_name": "bert-base-uncased",
            "pooling": cfg["pooling"],
            "fine_tune": cfg["fine_tune"],
            "dropout": cfg["dropout"],
            "batch_size": cfg["batch_size"],
            "text_col": text_col,
        }
        with open(save_dir / f"{save_name}_artifacts.pkl", "wb") as af:
            pickle.dump(artifacts, af)

    return result


def bert_run_single(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    train_config=None,
):
    """Run one end-to-end BERT experiment on a single split.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the column containing raw text.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        train_config: Optional training configuration overrides (see
            :func:`bert_train`).

    Returns:
        dict: The result dictionary returned by
        :func:`~modules.training_loop.training`.
    """
    return bert_train(
        train_df, valid_df, test_df, text_col,
        energy_model=energy_model,
        train_config=train_config,
    )


def bert_run_multiple(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    n=5,
    train_config=None,
):
    """Run multiple BERT training runs on the same split.

    :func:`bert_train` is called ``n`` times independently. Useful for
    estimating run-to-run variability due to random weight initialisation.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the column containing raw text.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        n: Number of training runs.
        train_config: Optional training configuration overrides (see
            :func:`bert_train`).

    Returns:
        list[dict]: A list of result dictionaries from each run.
    """
    return [
        bert_train(
            train_df, valid_df, test_df, text_col,
            energy_model=energy_model,
            train_config=train_config,
        )
        for _ in range(n)
    ]


def bert_hparam_search(
    train_df, valid_df, test_df, text_col,
    energy_model=True,
    n_trials=30,
    timeout=None,
):
    """Run Optuna hyperparameter search over BERT training hyperparameters.

    Searches over ``lr``, ``dropout``, ``pooling``, ``fine_tune``,
    ``batch_size``, and ``epochs``. Each trial trains a fresh model from
    scratch, so trials are independent.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the column containing raw text.
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

    def objective(trial):
        fine_tune  = trial.suggest_categorical("fine_tune", [True, False])
        pooling    = trial.suggest_categorical("pooling", ["cls", "mean"])
        dropout    = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs     = trial.suggest_int("epochs", 3, 15)
        lr_default = 2e-5 if fine_tune else 1e-4
        lr_low     = lr_default / 10
        lr_high    = lr_default * 10
        learning_rate = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)

        cfg = {
            "run_name": f"bert_hparam_trial_{trial.number}",
            "fine_tune": fine_tune,
            "pooling": pooling,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "verbose": False,
        }
        result = bert_train(
            train_df, valid_df, test_df, text_col,
            energy_model=energy_model,
            train_config=cfg,
        )
        return result["best_metric_value"]

    study = optuna.create_study(direction="maximize", study_name="bert_hparam_search")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study
