"""Experiment runner utilities for BiGRU classification."""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

from implementations.simple_bi_gru import BiGRUClassifier, build_bigru_dataloader
from modules.training_loop import training
from modules.encoding import LabelEncoder
from modules.encoding.vocab_encoder import VocabEncoder
from modules.encoding.sequence_encoder import SequenceEncoder
from modules import OneTextPreProcessor

_COLUMN_MAP_PATH = Path(__file__).parent.parent / "column_map.json"

_BIGRU_MODEL_TYPE = {
    "none": "bigru",
    "static": "bigru_safe_static",
    "contextual": "bigru_safe_context",
}


def _load_column_map():
    with open(_COLUMN_MAP_PATH) as f:
        return json.load(f)


def pre_process(data_df, text_col, keep_numbers=False, lemma_config={}):
    """Pre-process a dataset into the single-text format expected by BiGRU.

    Args:
        data_df: Input DataFrame to pre-process.
        text_col: Name of the text column to process.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization. If truthy,
            the preprocessor will lemmatize and produce lemma-token columns.

    Returns:
        pandas.DataFrame: The pre-processed DataFrame.
    """
    column_map = _load_column_map()
    proc = OneTextPreProcessor(
        keep_numbers=keep_numbers,
        column_map=column_map,
        drop_null=True,
        lemmatize=bool(lemma_config),
        lemma_config=lemma_config,
    )
    return proc.pre_process_df(data_df, text_col)


def pre_process_dataset(train_df, valid_df, test_df, text_col, keep_numbers=False, lemma_config={}):
    """Pre-process train/validation/test splits consistently.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to process.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: Processed
        ``(df_train, df_valid, df_test)`` DataFrames.
    """
    df_train = pre_process(train_df, text_col, keep_numbers, lemma_config)
    df_valid = pre_process(valid_df, text_col, keep_numbers, lemma_config)
    df_test  = pre_process(test_df,  text_col, keep_numbers, lemma_config)
    return df_train, df_valid, df_test


def bigru_encode(
    df_train, df_valid, df_test, text_col, lemma_config,
    max_len_percentile=95, min_vocab_freq=2,
):
    """Build vocabulary, encode sequences and labels for token-based BiGRU training.

    Fits a :class:`VocabEncoder` on the training split and determines
    ``max_len`` from the training-set length distribution. Both ``energy_type``
    and ``potential_damage`` labels are always encoded so the shared dataloader
    format is satisfied regardless of which target is used at training time.

    Args:
        df_train: Pre-processed training DataFrame.
        df_valid: Pre-processed validation DataFrame.
        df_test: Pre-processed test DataFrame.
        text_col: Base name of the original text column (used to derive the
            token column name).
        lemma_config: Lemmatization configuration. If truthy, uses
            ``f"{text_col}_tokens_lemma"``; otherwise uses
            ``f"{text_col}_tokens"``.
        max_len_percentile: Percentile of training-set sequence lengths used
            to cap padding. Defaults to 95 to avoid outlier-driven padding.
        min_vocab_freq: Minimum token frequency to include in the vocabulary.

    Returns:
        tuple: A 17-element tuple compatible with ``*encoded`` unpacking into
        :func:`bigru_train`:

        ``(vocab_enc, seq_enc, max_len,
        train_seqs, train_lengths, val_seqs, val_lengths, test_seqs, test_lengths,
        energy_enc, damage_enc,
        train_energy, val_energy, test_energy,
        train_damage, val_damage, test_damage)``
    """
    tokens_col = f"{text_col}_tokens_lemma" if lemma_config else f"{text_col}_tokens"

    train_tokenized_docs = df_train[tokens_col].tolist()
    val_tokenized_docs   = df_valid[tokens_col].tolist()
    test_tokenized_docs  = df_test[tokens_col].tolist()

    vocab_enc = VocabEncoder(min_freq=min_vocab_freq)
    vocab_enc.fit(train_tokenized_docs)

    max_len = int(np.percentile([len(d) for d in train_tokenized_docs], max_len_percentile))
    seq_enc = SequenceEncoder(vocab_enc, max_length=max_len)

    def _encode_seqs(docs):
        seqs    = torch.tensor(seq_enc.encode_sequences(docs), dtype=torch.long)
        lengths = torch.tensor([min(len(d), max_len) for d in docs], dtype=torch.long)
        return seqs, lengths

    train_seqs, train_lengths = _encode_seqs(train_tokenized_docs)
    val_seqs,   val_lengths   = _encode_seqs(val_tokenized_docs)
    test_seqs,  test_lengths  = _encode_seqs(test_tokenized_docs)

    energy_enc = LabelEncoder()
    energy_enc.fit(df_train["energy_type"].tolist())
    train_energy = torch.tensor(energy_enc.encode_many(df_train["energy_type"].tolist()))
    val_energy   = torch.tensor(energy_enc.encode_many(df_valid["energy_type"].tolist()))
    test_energy  = torch.tensor(energy_enc.encode_many(df_test["energy_type"].tolist()))

    damage_enc = LabelEncoder()
    damage_enc.fit(df_train["potential_damage"].tolist())
    train_damage = torch.tensor(damage_enc.encode_many(df_train["potential_damage"].tolist()))
    val_damage   = torch.tensor(damage_enc.encode_many(df_valid["potential_damage"].tolist()))
    test_damage  = torch.tensor(damage_enc.encode_many(df_test["potential_damage"].tolist()))

    return (
        vocab_enc, seq_enc, max_len,
        train_seqs,   train_lengths,
        val_seqs,     val_lengths,
        test_seqs,    test_lengths,
        energy_enc,   damage_enc,
        train_energy, val_energy,   test_energy,
        train_damage, val_damage,   test_damage,
    )


def bigru_contextual_encode(df_train, df_valid, df_test, text_col, embedding_model_name="bert-base-uncased"):
    """Encode all splits with contextual BERT embeddings and encode both label sets.

    Produces the same 17-element tuple as :func:`bigru_encode` (with
    ``vocab_enc``, ``seq_enc``, and ``max_len`` set to ``None``) so the result
    can be unpacked directly into :func:`bigru_train`.

    Args:
        df_train: Pre-processed training DataFrame (must contain ``text_col``
            and both label columns).
        df_valid: Pre-processed validation DataFrame.
        df_test: Pre-processed test DataFrame.
        text_col: Column name of the raw text used for BERT encoding.
        embedding_model_name: HuggingFace model identifier for the BERT encoder.

    Returns:
        tuple: Same 17-element structure as :func:`bigru_encode`.
    """
    from modules.embedding.safety_bert_static import get_contextual_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _embed_and_pack(texts):
        embs    = get_contextual_embeddings(texts, model_name=embedding_model_name, device=device)
        lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
        padded  = pad_sequence(embs, batch_first=True)   # (N, max_seq_len, hidden_dim)
        return padded, lengths

    train_seqs, train_lengths = _embed_and_pack(df_train[text_col])
    val_seqs,   val_lengths   = _embed_and_pack(df_valid[text_col])
    test_seqs,  test_lengths  = _embed_and_pack(df_test[text_col])

    energy_enc = LabelEncoder()
    energy_enc.fit(df_train["energy_type"].tolist())
    train_energy = torch.tensor(energy_enc.encode_many(df_train["energy_type"].tolist()))
    val_energy   = torch.tensor(energy_enc.encode_many(df_valid["energy_type"].tolist()))
    test_energy  = torch.tensor(energy_enc.encode_many(df_test["energy_type"].tolist()))

    damage_enc = LabelEncoder()
    damage_enc.fit(df_train["potential_damage"].tolist())
    train_damage = torch.tensor(damage_enc.encode_many(df_train["potential_damage"].tolist()))
    val_damage   = torch.tensor(damage_enc.encode_many(df_valid["potential_damage"].tolist()))
    test_damage  = torch.tensor(damage_enc.encode_many(df_test["potential_damage"].tolist()))

    return (
        None, None, None,             # vocab_enc, seq_enc, max_len — unused in contextual mode
        train_seqs,   train_lengths,
        val_seqs,     val_lengths,
        test_seqs,    test_lengths,
        energy_enc,   damage_enc,
        train_energy, val_energy,   test_energy,
        train_damage, val_damage,   test_damage,
    )


_BIGRU_TRAIN_DEFAULTS = {
    "epochs": 50,
    "patience": 10,
    "best_metric": "f1_macro",
    "criterion_type": "focal",
    "need_length": True,
    "scheduler_step_per_batch": False,
}


def bigru_train(
    vocab_enc, seq_enc, max_len,
    train_seqs,   train_lengths,
    val_seqs,     val_lengths,
    test_seqs,    test_lengths,
    energy_enc,   damage_enc,
    train_energy, val_energy,   test_energy,
    train_damage, val_damage,   test_damage,
    energy_model=True,
    train_config=None,
    requirements={},
    artifact_extras=None,
):
    """Train a BiGRU classifier.

    Accepts the tuple produced by :func:`bigru_encode` or
    :func:`bigru_contextual_encode` (unpack with ``*encoded``).

    Three embedding modes are supported via ``train_config["embedding_type"]``:

    - ``"none"`` *(default)* — learnable embeddings trained from scratch.
    - ``"static"`` — SafetyBERT (or another HuggingFace BERT model) token
      embeddings are extracted once and injected into the embedding layer.
    - ``"contextual"`` — sequences are pre-computed contextual BERT embeddings
      (float tensors); the embedding layer is bypassed entirely.

    Args:
        vocab_enc: Fitted :class:`VocabEncoder` (``None`` for contextual mode).
        seq_enc: Fitted :class:`SequenceEncoder` (``None`` for contextual mode).
        max_len: Sequence cap used during encoding (``None`` for contextual mode).
        train_seqs: Padded training sequences (Long tensor for token-based modes,
            Float tensor for contextual mode).
        train_lengths: Unpadded lengths for the training split.
        val_seqs: Padded validation sequences.
        val_lengths: Unpadded lengths for the validation split.
        test_seqs: Padded test sequences.
        test_lengths: Unpadded lengths for the test split.
        energy_enc: Fitted :class:`LabelEncoder` for ``energy_type``.
        damage_enc: Fitted :class:`LabelEncoder` for ``potential_damage``.
        train_energy: Encoded energy-type training labels.
        val_energy: Encoded energy-type validation labels.
        test_energy: Encoded energy-type test labels.
        train_damage: Encoded potential-damage training labels.
        val_damage: Encoded potential-damage validation labels.
        test_damage: Encoded potential-damage test labels.
        energy_model: If ``True``, optimise for ``energy_type``; otherwise for
            ``potential_damage``.
        train_config: Optional dict of overrides. Special keys (resolved before
            forwarding to :func:`~modules.training_loop.training`):

            - ``embedding_type``: ``"none"`` | ``"static"`` | ``"contextual"``
            - ``embedding_dim``: int (default 128 for ``"none"``, 768 for BERT)
            - ``hidden_dim``: int (default 128)
            - ``dropout_prob``: float (default 0.3)
            - ``freeze_emb``: bool (default ``False``)
            - ``embedding_model_name``: HuggingFace model identifier used in
              ``"static"`` mode (default ``"adanish91/safetybert"``)
            - ``optimizer_fn``: ``Callable[[nn.Module], Optimizer]``
            - ``scheduler_fn``: ``Callable[[Optimizer], LRScheduler]``

        requirements: Optional client performance requirements dict, defaults to {}.
            Pass ``None`` to disable the check. Keys:

            - ``confidence_threshold``: ``{"high": float, "medium": float}``
            - ``high_threshold``: min fraction of predictions in the high-confidence tier
            - ``fatal_accuracy``: min recall on true fatal-class samples
            - ``f1_target``: ``{class_index: min_f1}``
        artifact_extras: Optional dict passed from :func:`bigru_run_single` to persist
            ``*_artifacts.pkl`` for inference preprocessing.

    Returns:
        dict: The result dictionary returned by
        :func:`~modules.training_loop.training`.
    """
    cfg = {**_BIGRU_TRAIN_DEFAULTS, **(train_config or {})}

    batch_size = int(cfg.pop("batch_size", 32))
    embedding_type       = cfg.pop("embedding_type", "none")
    hidden_dim           = cfg.pop("hidden_dim", 128)
    dropout_prob         = cfg.pop("dropout_prob", 0.3)
    freeze_emb           = cfg.pop("freeze_emb", False)
    embedding_model_name = cfg.pop("embedding_model_name", "adanish91/safetybert")

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    label_enc   = energy_enc if energy_model else damage_enc
    num_classes = label_enc.num_classes

    if embedding_type == "none":
        embedding_dim = cfg.pop("embedding_dim", 128)
        model = BiGRUClassifier(
            vocab_size=vocab_enc.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            freeze_emb=freeze_emb,
        )

    elif embedding_type == "static":
        from modules.embedding.safety_bert_static import get_safety_bert_embedding_matrix
        embedding_dim = cfg.pop("embedding_dim", 768)
        bert_matrix = get_safety_bert_embedding_matrix(
            vocab=vocab_enc.token_to_id,
            model_name=embedding_model_name,
            device=device,
            verbose=False,
        )
        model = BiGRUClassifier(
            vocab_size=vocab_enc.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            emb_table=bert_matrix.cpu().numpy(),
            dropout_prob=dropout_prob,
            freeze_emb=freeze_emb,
        )

    elif embedding_type == "contextual":
        embedding_dim = cfg.pop("embedding_dim", 768)
        # vocab_size=1 is a placeholder; the embedding layer is bypassed for float input
        model = BiGRUClassifier(
            vocab_size=1,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            freeze_emb=False,
        )

    else:
        raise ValueError(
            f"Unknown embedding_type={embedding_type!r}. "
            "Expected 'none', 'static', or 'contextual'."
        )

    model = model.to(device)

    train_dl = build_bigru_dataloader(
        train_seqs, train_lengths, train_energy, train_damage, batch_size=batch_size
    )
    val_dl   = build_bigru_dataloader(
        val_seqs, val_lengths, val_energy, val_damage, batch_size=batch_size, shuffle=False
    )
    test_dl  = build_bigru_dataloader(
        test_seqs, test_lengths, test_energy, test_damage, batch_size=batch_size, shuffle=False
    )

    optimizer_fn = cfg.pop("optimizer_fn", None)
    scheduler_fn = cfg.pop("scheduler_fn", None)

    optimiser = optimizer_fn(model) if optimizer_fn is not None else None
    scheduler = scheduler_fn(optimiser) if (scheduler_fn is not None and optimiser is not None) else None

    result = training(
        model_type=_BIGRU_MODEL_TYPE[embedding_type],
        model=model,
        train_dl=train_dl,
        valid_dl=val_dl,
        test_dl=test_dl,
        device=device,
        energy_model=energy_model,
        num_classes=num_classes,
        optimiser=optimiser,
        scheduler=scheduler,
        requirements=requirements,
        class_dict=label_enc.id_to_label,
        **cfg,
    )

    if (
        artifact_extras is not None
        and result.get("best_model_state_dict") is not None
        and result["config"].get("save")
    ):
        save_dir = Path(result["config"]["save_dir"])
        save_name = result["config"]["save_name"]
        embedding_dim_saved = getattr(model.word_embeddings, "embedding_dim", embedding_dim)

        artifacts = {
            "vocab_enc": vocab_enc,
            "seq_enc": seq_enc,
            "max_len": max_len,
            "energy_enc": energy_enc,
            "damage_enc": damage_enc,
            "embedding_type": embedding_type,
            "embedding_model_name": embedding_model_name,
            "energy_model": energy_model,
            "hidden_dim": hidden_dim,
            "dropout_prob": dropout_prob,
            "freeze_emb": freeze_emb,
            "embedding_dim": int(embedding_dim_saved),
            "batch_size": batch_size,
            "text_col": artifact_extras.get("text_col"),
            "lemma_config": artifact_extras.get("lemma_config"),
            "keep_numbers": artifact_extras.get("keep_numbers", False),
        }
        with open(save_dir / f"{save_name}_artifacts.pkl", "wb") as af:
            pickle.dump(artifacts, af)

    return result


def bigru_run_single(
    train_df, valid_df, test_df, text_col,
    energy_model=True, keep_numbers=False, lemma_config={},
    train_config=None,
    requirements={},
):
    """Run one end-to-end BiGRU experiment on a single split.

    Pre-processes the splits, encodes sequences and labels, then trains one
    BiGRU model. Supports all three embedding modes via
    ``train_config["embedding_type"]``.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to use.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.
        train_config: Optional training configuration overrides (see
            :func:`bigru_train`).
        requirements: Optional client performance requirements dict (see
            :func:`bigru_train`).

    Returns:
        dict: The result dictionary returned by
        :func:`~modules.training_loop.training`.
    """
    embedding_type = (train_config or {}).get("embedding_type", "none")

    df_train, df_valid, df_test = pre_process_dataset(
        train_df, valid_df, test_df, text_col, keep_numbers, lemma_config
    )

    if embedding_type == "contextual":
        embedding_model_name = (train_config or {}).get("embedding_model_name", "bert-base-uncased")
        encoded = bigru_contextual_encode(
            df_train, df_valid, df_test, text_col,
            embedding_model_name=embedding_model_name,
        )
    else:
        encoded = bigru_encode(df_train, df_valid, df_test, text_col, lemma_config)

    artifact_extras = {
        "text_col": text_col,
        "lemma_config": lemma_config,
        "keep_numbers": keep_numbers,
    }
    return bigru_train(
        *encoded,
        energy_model=energy_model,
        train_config=train_config,
        requirements=requirements,
        artifact_extras=artifact_extras,
    )


def bigru_run_multiple(
    train_df, valid_df, test_df, text_col,
    energy_model=True, keep_numbers=False, lemma_config={},
    n=5, train_config=None, requirements={},
):
    """Run multiple BiGRU training runs on the same processed split.

    Data is pre-processed and encoded once; :func:`bigru_train` is called ``n``
    times. Useful for estimating run-to-run variability due to random weight
    initialisation.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to use.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.
        n: Number of training runs.
        train_config: Optional training configuration overrides (see
            :func:`bigru_train`).
        requirements: Optional client performance requirements dict (see
            :func:`bigru_train`).

    Returns:
        list[dict]: A list of result dictionaries from each run.
    """
    embedding_type = (train_config or {}).get("embedding_type", "none")

    df_train, df_valid, df_test = pre_process_dataset(
        train_df, valid_df, test_df, text_col, keep_numbers, lemma_config
    )

    if embedding_type == "contextual":
        embedding_model_name = (train_config or {}).get("embedding_model_name", "bert-base-uncased")
        encoded = bigru_contextual_encode(
            df_train, df_valid, df_test, text_col,
            embedding_model_name=embedding_model_name,
        )
    else:
        encoded = bigru_encode(df_train, df_valid, df_test, text_col, lemma_config)

    return [
        bigru_train(*encoded, energy_model=energy_model, train_config=train_config, requirements=requirements)
        for _ in range(n)
    ]


def bigru_hparam_search(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config={}, energy_model=True,
    n_trials=40, timeout=None, requirements={},
):
    """Run Optuna hyperparameter search over lr, hidden_dim, embedding_dim, and scheduler.

    Pre-processing and vocabulary encoding are done once before the study
    starts. Each trial trains a fresh model, so trials are independent.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to use.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.
        energy_model: If ``True``, predict ``energy_type``; otherwise predict
            ``potential_damage``.
        n_trials: Maximum number of Optuna trials.
        timeout: Stop after this many seconds regardless of ``n_trials``
            (``None`` = no limit).
        requirements: Optional client performance requirements dict (see
            :func:`bigru_train`).

    Returns:
        optuna.Study: Inspect with ``study.best_params``, ``study.best_value``,
        and ``optuna.visualization`` helpers.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_train, df_valid, df_test = pre_process_dataset(
        train_df, valid_df, test_df, text_col, keep_numbers, lemma_config
    )
    encoded = bigru_encode(df_train, df_valid, df_test, text_col, lemma_config)

    def objective(trial):
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim    = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        epochs        = trial.suggest_int("epochs", 20, 100, step=10)
        scheduler     = trial.suggest_categorical("scheduler", ["cosine", "cosine_warmup", "step", "none"])

        def optimizer_fn(model):
            return torch.optim.Adam(model.parameters(), lr=lr)

        def scheduler_fn(opt):
            if scheduler == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
            if scheduler == "cosine_warmup":
                warmup = max(1, epochs // 10)
                return torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
                        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=1e-6),
                    ],
                    milestones=[warmup],
                )
            if scheduler == "step":
                return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
            return False

        cfg = {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "optimizer_fn": optimizer_fn,
            "scheduler_fn": scheduler_fn,
            "scheduler_step_per_batch": False,
            "patience": 12,
            "best_metric": "f1_macro",
            "save": True,
            "log_leaderboard": True,
            "verbose": False,
        }
        result = bigru_train(*encoded, energy_model=energy_model, train_config=cfg, requirements=requirements)
        return result["best_metric_value"]

    study = optuna.create_study(direction="maximize", study_name="bigru_hparam_search")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study
