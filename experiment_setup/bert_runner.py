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
    "model_type": "bert-base-uncased",                 # remove from config after refactor
    "tokenizer_name": None,
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
            "model_name": cfg["model_type"],
            "tokenizer_name": cfg.get("tokenizer_name") or cfg["model_type"],
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


def bert_continue_train(
    bundle,
    train_df,
    valid_df,
    test_df,
    text_col,
    train_config=None,
):
    """Continue fine-tuning a saved BERT classifier checkpoint."""
    import pickle
    from pathlib import Path

    import torch
    import torch.optim as optim

    from modules.data_loader.bert_loader import df_to_bert_dataloader
    from modules.embedding.bert_config import BertEmbeddingConfig
    from modules.embedding.bert_tokenizer import BertTokenizerWrapper
    from modules.embedding.bert_embedding import BertEmbeddingBackend
    from modules.models.bert_classifier import BertClassifier
    from modules.training_loop import training
    from implementations.bert_transformer import get_best_available_device, build_class_weights

    cfg = {**_BERT_DEFAULTS, **(train_config or {})}

    artifacts = bundle["artifacts"]
    old_config = bundle.get("config", {})

    energy_model = bool(artifacts.get("energy_model", bundle.get("energy_model", False)))
    label_col = "energy_type" if energy_model else "potential_damage"
    target_type = "energy" if energy_model else "risk"

    label_enc = bundle["label_enc"]

    def encode_existing(df):
        out = df.copy()
        out[label_col] = label_enc.encode_many(out[label_col].astype(str).tolist())
        return out

    train_df = encode_existing(train_df)
    valid_df = encode_existing(valid_df)
    test_df = encode_existing(test_df)

    class_names = [label_enc.id_to_label[i] for i in range(label_enc.num_classes)]

    device = get_best_available_device()

    model_name = artifacts.get("model_name", old_config.get("model_name", "bert-base-uncased"))
    tokenizer_name = artifacts.get("tokenizer_name", old_config.get("tokenizer_name", model_name))

    max_length = int(cfg.get("max_length", artifacts.get("max_length", 160)))
    pooling = cfg.get("pooling", artifacts.get("pooling", "mean"))
    fine_tune = bool(cfg.get("fine_tune", artifacts.get("fine_tune", True)))
    dropout = float(cfg.get("dropout", artifacts.get("dropout", 0.2)))
    batch_size = int(cfg.get("batch_size", artifacts.get("batch_size", 8)))
    weight_decay = float(cfg.get("weight_decay", old_config.get("weight_decay", 0.01)))
    threshold = float(cfg.get("threshold", old_config.get("threshold", 0.8)))
    epochs = int(cfg.get("epochs", 3))
    patience = int(cfg.get("patience", old_config.get("patience", 2)))

    learning_rate = cfg.get("learning_rate")
    if learning_rate is None:
        learning_rate = 1e-5 if fine_tune else 1e-4

    bert_config = BertEmbeddingConfig(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dropout=float(cfg.get("embedding_dropout", 0.1)),
        pooling=pooling,
        fine_tune=fine_tune,
    )

    tokenizer_wrapper = BertTokenizerWrapper(bert_config)

    train_dl = df_to_bert_dataloader(
        train_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_dl = df_to_bert_dataloader(
        valid_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dl = df_to_bert_dataloader(
        test_df,
        text_col=text_col,
        label_col=label_col,
        tokenizer_wrapper=tokenizer_wrapper,
        batch_size=batch_size,
        shuffle=False,
    )

    embedding_backend = BertEmbeddingBackend(bert_config)

    model = BertClassifier(
        embedding_backend=embedding_backend,
        num_classes=label_enc.num_classes,
        dropout=dropout,
    ).to(device)

    # Load the previously exported checkpoint into the newly constructed model.
    model.load_state_dict(bundle["model"].state_dict())

    optimiser = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion_weights = None
    if bool(cfg.get("use_class_weights", False)):
        criterion_weights = build_class_weights(
            train_df=train_df,
            label_col=label_col,
            device=device,
        )

    run_name = cfg.get(
        "run_name",
        f"{old_config.get('run_name', old_config.get('save_name', 'bert'))}_continue",
    )

    result = training(
        model=model,
        energy_model=energy_model,
        model_type=old_config.get("model_type", "BERT"),
        need_length=False,
        optimiser=optimiser,
        scheduler=cfg.get("scheduler_config"),
        scheduler_step_per_batch=(
            cfg.get("scheduler_config", {}) or {}
        ).get("step_per_batch", False),
        criterion_type="cross_entropy",
        criterion_weights=criterion_weights,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        epochs=epochs,
        patience=patience,
        num_classes=label_enc.num_classes,
        class_dict=label_enc.id_to_label,
        clip_grad_max_norm=1.0,
        best_metric=cfg.get("best_metric", "f1_macro"),
        threshold=threshold,
        temperature=float(old_config.get("temperature", 1.0)),
        use_temperature=bool(old_config.get("use_temperature", False)),
        parameters={
            "retrain_mode": "continue",
            "base_model_dir": old_config.get("save_dir"),
            "model_type": old_config.get("model_type", "BERT"),
            "target_type": target_type,
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "pooling": pooling,
            "fine_tune": fine_tune,
            "batch_size": batch_size,
            "label_col": label_col,
            "text_col": text_col,
            "learning_rate": learning_rate,
            "classifier_dropout": dropout,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "threshold": threshold,
            "device": str(device),
        },
        device=device,
        save=True,
        parent_dir=cfg.get("parent_dir", "trained_models"),
        run_name=run_name,
        verbose=cfg.get("verbose", True),
        extra_config={
            "class_names": class_names,
            "label_col": label_col,
            "text_col": text_col,
            "target_type": target_type,
            "model_type": old_config.get("model_type", "BERT"),
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "pooling": pooling,
            "fine_tune": fine_tune,
            "base_model_dir": old_config.get("save_dir"),
            "retrain_mode": "continue",
        },
    )

    if result.get("best_model_state_dict") is not None and result["config"].get("save"):
        save_dir = Path(result["config"]["save_dir"])
        save_name = result["config"]["save_name"]

        updated_artifacts = {
            **artifacts,
            "energy_model": energy_model,
            "label_enc": label_enc,
            "max_length": max_length,
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "pooling": pooling,
            "fine_tune": fine_tune,
            "dropout": dropout,
            "batch_size": batch_size,
            "text_col": text_col,
            "base_model_dir": old_config.get("save_dir"),
            "retrain_mode": "continue",
        }

        with open(save_dir / f"{save_name}_artifacts.pkl", "wb") as af:
            pickle.dump(updated_artifacts, af)

    return result