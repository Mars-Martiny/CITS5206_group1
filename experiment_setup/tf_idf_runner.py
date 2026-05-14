"""Experiment runner utilities for TF-IDF classification."""

import json
import pickle
import torch
from pathlib import Path

from implementations.tf_idf import TFIDFClassifier, TFIDFVectorizer, build_tfidf_dataloader
from modules.training_loop import training
from modules.encoding import LabelEncoder
from modules import OneTextPreProcessor

_COLUMN_MAP_PATH = Path(__file__).parent.parent / "column_map.json"


def _load_column_map():
    """Load the configured column name mapping.

    Returns:
        dict: Mapping loaded from `column_map.json`.
    """
    with open(_COLUMN_MAP_PATH) as f:
        return json.load(f)


def pre_process(data_df, text_col, keep_numbers, lemma_config={}):
    """Pre-process a dataset into the single-text format expected by TF-IDF.

    This uses `OneTextPreProcessor` with the repository's `column_map.json` and
    returns the processed DataFrame, including token columns.

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
    oneTextPreProcessor = OneTextPreProcessor(
        keep_numbers=keep_numbers,
        column_map=column_map,
        drop_null=True,
        lemmatize=bool(lemma_config),
        lemma_config=lemma_config,
    )
    return oneTextPreProcessor.pre_process_df(data_df, text_col)


def pre_process_dataset(train_df, valid_df, test_df, text_col, keep_numbers, lemma_config):
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
        `(df_train, df_valid, df_test)` DataFrames.
    """
    df_train = pre_process(train_df, text_col, keep_numbers, lemma_config)
    df_valid = pre_process(valid_df, text_col, keep_numbers, lemma_config)
    df_test  = pre_process(test_df,  text_col, keep_numbers, lemma_config)
    return df_train, df_valid, df_test


def tf_idf_encode(df_train, df_valid, df_test, text_col, lemma_config, energy_model=True):
    """Prepare tokenized documents and encoded labels for TF-IDF training.

    This selects the label column based on `energy_model` and selects the token
    column based on whether lemmatization is enabled via `lemma_config`.

    Args:
        df_train: Pre-processed training DataFrame.
        df_valid: Pre-processed validation DataFrame.
        df_test: Pre-processed test DataFrame.
        text_col: Base name of the original text column (used to derive the
            token column name).
        lemma_config: Lemmatization configuration. If truthy, uses
            `f"{text_col}_tokens_lemma"`; otherwise uses `f"{text_col}_tokens"`.
        energy_model: If True, use the `energy_type` label; otherwise use
            `potential_damage`.

    Returns:
        tuple: A tuple containing:

        - label_col (str): Chosen label column name.
        - train_tokenized_docs (list): Tokenized documents for training.
        - val_tokenized_docs (list): Tokenized documents for validation.
        - test_tokenized_docs (list): Tokenized documents for testing.
        - label_enc (LabelEncoder): Fitted label encoder.
        - train_labels (torch.Tensor): Encoded training labels.
        - val_labels (torch.Tensor): Encoded validation labels.
        - test_labels (torch.Tensor): Encoded test labels.
        - energy_model (bool): Passed through from the input.
    """
    label_col  = "energy_type" if energy_model else "potential_damage"
    tokens_col = f"{text_col}_tokens_lemma" if lemma_config else f"{text_col}_tokens"

    train_tokenized_docs = df_train[tokens_col].tolist()
    val_tokenized_docs   = df_valid[tokens_col].tolist()
    test_tokenized_docs  = df_test[tokens_col].tolist()

    label_enc = LabelEncoder()
    label_enc.fit(df_train[label_col].tolist())

    train_labels = torch.tensor(label_enc.encode_many(df_train[label_col].tolist()))
    val_labels   = torch.tensor(label_enc.encode_many(df_valid[label_col].tolist()))
    test_labels  = torch.tensor(label_enc.encode_many(df_test[label_col].tolist()))

    return (
        label_col,
        train_tokenized_docs,
        val_tokenized_docs,
        test_tokenized_docs,
        label_enc,
        train_labels,
        val_labels,
        test_labels,
        energy_model,
    )


_TFIDF_TRAIN_DEFAULTS = {
    "epochs": 100,
    "patience": 12,
    "best_metric": "f1_macro",
    "criterion_weights": None,
    "need_length": False,
    "scheduler_step_per_batch": False,
}


def tf_idf_train(
    label_col,
    train_tokenized_docs,
    val_tokenized_docs,
    test_tokenized_docs,
    label_enc,
    train_labels,
    val_labels,
    test_labels,
    energy_model,
    train_config=None,
    requirements={},
    artifact_extras=None,
):
    """Train a TF-IDF classifier.

    This fits a `TFIDFVectorizer` on the training documents, transforms all
    splits to TF-IDF vectors, builds dataloaders, and delegates optimization and
    evaluation to `modules.training_loop.training()`.

    Args:
        label_col: Name of the label column (informational; labels are already
            provided as tensors).
        train_tokenized_docs: Tokenized training documents (list of token
            sequences).
        val_tokenized_docs: Tokenized validation documents (list of token
            sequences).
        test_tokenized_docs: Tokenized test documents (list of token sequences).
        label_enc: Fitted `LabelEncoder` for the labels.
        train_labels: Encoded training labels as a 1D `torch.Tensor`.
        val_labels: Encoded validation labels as a 1D `torch.Tensor`.
        test_labels: Encoded test labels as a 1D `torch.Tensor`.
        energy_model: Whether the task is energy type classification (True) or
            potential damage classification (False). Passed through to
            `training()`.
        train_config: Optional dict of overrides passed to ``training()``.
            Two special keys are resolved before forwarding:

            - ``optimizer_fn``: ``Callable[[nn.Module], Optimizer]`` — receives
              the model and returns an instantiated optimizer.
            - ``scheduler_fn``: ``Callable[[Optimizer], LRScheduler]`` — receives
              the optimizer and returns an instantiated scheduler.

            All other keys are passed verbatim as keyword arguments to
            ``training()``, overriding the defaults in ``_TFIDF_TRAIN_DEFAULTS``.
        requirements:  Optional client performance requirements dict, defaults to {}. 
            Pass None to disable check. Keys:
            - confidence_threshold: {"high": float, "medium": float} (values >1 treated as %)
            - high_threshold: min fraction of predictions in high-confidence tier (default 0.70)
            - fatal_accuracy: min recall on true fatal-class samples (default 0.95)
            - f1_target: {class_index: min_f1} — use 0.0 to mark a class as having no target
        artifact_extras: Optional dict with ``text_col``, ``lemma_config``, ``keep_numbers`` for saving
            ``*_artifacts.pkl`` after training when ``artifact_extras`` is not None.

    Returns:
        dict: The result dictionary returned by `training()`.
    """
    cfg = {**_TFIDF_TRAIN_DEFAULTS, **(train_config or {})}

    batch_size = int(cfg.pop("batch_size", 32))
    hidden_dim_param = cfg.pop("hidden_dim", 256)

    vectorizer = TFIDFVectorizer().fit(train_tokenized_docs)

    feature_representation = cfg.pop("feature_representation", "tfidf")
    embedding_model_name = cfg.pop("embedding_model_name", "adanish91/safetybert")

    if feature_representation == "tfidf":
        train_vecs = vectorizer.transform(train_tokenized_docs)
        val_vecs   = vectorizer.transform(val_tokenized_docs)
        test_vecs  = vectorizer.transform(test_tokenized_docs)
        input_dim = len(vectorizer.vocab)
    elif feature_representation == "tfidf_embed_avg":
        print("Use SafetyBERT on raw")
        # TF-IDF-weighted average of SafetyBERT static input embeddings.
        # This keeps the rest of the training pipeline unchanged (dense vectors
        # go through the same dataloader and the same TFIDFClassifier).
        from modules.embedding.safety_bert_static import get_safety_bert_embedding_matrix

        E = get_safety_bert_embedding_matrix(vectorizer.vocab, model_name=embedding_model_name, verbose=False)
        train_vecs = vectorizer.transform_weighted_average_embeddings(train_tokenized_docs, embedding_matrix=E)
        val_vecs   = vectorizer.transform_weighted_average_embeddings(val_tokenized_docs,   embedding_matrix=E)
        test_vecs  = vectorizer.transform_weighted_average_embeddings(test_tokenized_docs,  embedding_matrix=E)
        input_dim = int(train_vecs.shape[1])
        print("len(vectorizer.vocab)", len(vectorizer.vocab))
        print("E.shape", E.shape) 
        print("E.sum", E.sum()) 
        print("E.std", E.std())
        print("E.train_vecs[:5]", train_vecs[:5])
        print("Train vecs dim", train_vecs.std(dim=0).mean())
    else:
        raise ValueError(
            f"Unknown feature_representation={feature_representation!r}. "
            "Expected 'tfidf' or 'tfidf_embed_avg'."
        )

    train_dl = build_tfidf_dataloader(train_vecs, train_labels, batch_size=batch_size)
    val_dl   = build_tfidf_dataloader(val_vecs, val_labels, batch_size=batch_size, shuffle=False)
    test_dl  = build_tfidf_dataloader(test_vecs, test_labels, batch_size=batch_size, shuffle=False)

    num_classes = label_enc.num_classes
    model = TFIDFClassifier(
        vocab_size=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim_param,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    
    optimizer_fn  = cfg.pop("optimizer_fn", None)
    scheduler_config = cfg.pop("scheduler_config", None)

    optimiser = optimizer_fn(model) if optimizer_fn is not None else None
    scheduler = scheduler_config

    result = training(
        model_type="tf_idf",
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
        artifacts = {
            "vectorizer": vectorizer,
            "label_enc": label_enc,
            "energy_model": energy_model,
            "feature_representation": feature_representation,
            "embedding_model_name": embedding_model_name,
            "hidden_dim": hidden_dim_param,
            "batch_size": batch_size,
            "text_col": artifact_extras.get("text_col"),
            "lemma_config": artifact_extras.get("lemma_config"),
            "keep_numbers": artifact_extras.get("keep_numbers", False),
        }
        artifact_path = save_dir / f"{save_name}_artifacts.pkl"
        with open(artifact_path, "wb") as af:
            pickle.dump(artifacts, af)

    return result


def tf_idf_run_single(
    train_df, valid_df, test_df, text_col,
    energy_model=True, keep_numbers=False, lemma_config={},
    train_config=None,
    requirements={}
):
    """Run one end-to-end TF-IDF experiment on a single split.

    This pre-processes the splits, encodes labels, and trains a TF-IDF model
    once.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to use.
        energy_model: If True, predict `energy_type`; otherwise predict
            `potential_damage`.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.
        train_config: Optional training configuration overrides (see
            `tf_idf_train`).
        requirements:  Optional client performance requirements dict, defaults to {}. 
            Pass None to disable check. Keys:
            - confidence_threshold: {"high": float, "medium": float} (values >1 treated as %)
            - high_threshold: min fraction of predictions in high-confidence tier (default 0.70)
            - fatal_accuracy: min recall on true fatal-class samples (default 0.95)
            - f1_target: {class_index: min_f1} — use 0.0 to mark a class as having no target

    Returns:
        dict: The result dictionary returned by `training()`.
    """
    df_train, df_valid, df_test = pre_process_dataset(
        train_df, valid_df, test_df, text_col, keep_numbers, lemma_config
    )
    encoded = tf_idf_encode(df_train, df_valid, df_test, text_col, lemma_config, energy_model)
    artifact_extras = {"text_col": text_col, "lemma_config": lemma_config, "keep_numbers": keep_numbers}
    return tf_idf_train(
        *encoded,
        train_config=train_config,
        requirements=requirements,
        artifact_extras=artifact_extras,
    )


def tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config={}, energy_model=True, n=5,
    train_config=None,
    requirements = {}
):
    """Run multiple TF-IDF training runs on the same processed split.

    The data is pre-processed and encoded once, then `tf_idf_train()` is called
    `n` times. This is useful for estimating run-to-run variability.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to use.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.
        energy_model: If True, predict `energy_type`; otherwise predict
            `potential_damage`.
        n: Number of runs to execute.
        train_config: Optional training configuration overrides (see
            `tf_idf_train`).
        requirements:  Optional client performance requirements dict, defaults to {}. 
            Pass None to disable check. Keys:
            - confidence_threshold: {"high": float, "medium": float} (values >1 treated as %)
            - high_threshold: min fraction of predictions in high-confidence tier (default 0.70)
            - fatal_accuracy: min recall on true fatal-class samples (default 0.95)
            - f1_target: {class_index: min_f1} — use 0.0 to mark a class as having no target
    Returns:
        list[dict]: A list of result dictionaries returned by `training()`.
    """
    df_train, df_valid, df_test = pre_process_dataset(
        train_df, valid_df, test_df, text_col, keep_numbers, lemma_config
    )
    encoded = tf_idf_encode(df_train, df_valid, df_test, text_col, lemma_config, energy_model)
    return [tf_idf_train(*encoded, train_config=train_config, requirements=requirements) for _ in range(n)]


def tf_idf_hparam_search(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config={}, energy_model=True,
    n_trials=40, timeout=None, requirements= {}
):
    """Run Optuna hyperparameter search over lr, hidden_dim, epochs, and scheduler.

    Pre-processing and tokenization are done once before the study starts.
    Each trial fits a fresh vectorizer and trains a new model, so trials are
    independent. Artifacts and leaderboard logging are suppressed per trial.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        text_col: Name of the text column to use.
        keep_numbers: Whether to keep numeric tokens during preprocessing.
        lemma_config: Optional configuration enabling lemmatization.
        energy_model: If True, predict `energy_type`; otherwise predict
            `potential_damage`.
        n_trials:  Maximum number of Optuna trials.
        timeout:   Stop after this many seconds regardless of n_trials (None = no limit).
        requirements:  Optional client performance requirements dict, defaults to {}. 
            Pass None to disable check. Keys:
            - confidence_threshold: {"high": float, "medium": float} (values >1 treated as %)
            - high_threshold: min fraction of predictions in high-confidence tier (default 0.70)
            - fatal_accuracy: min recall on true fatal-class samples (default 0.95)
            - f1_target: {class_index: min_f1} — use 0.0 to mark a class as having no target
    Returns:
        optuna.Study — inspect with study.best_params, study.best_value,
        and optuna.visualization helpers.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-process once — spacy lemmatisation is the bottleneck
    df_train, df_valid, df_test = pre_process_dataset(
        train_df, valid_df, test_df, text_col, keep_numbers, lemma_config
    )
    encoded = tf_idf_encode(df_train, df_valid, df_test, text_col, lemma_config, energy_model)

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        epochs     = trial.suggest_int("epochs", 30, 150, step=10)
        scheduler  = trial.suggest_categorical("scheduler", ["cosine", "cosine_warmup", "step", "none"])

        def optimizer_fn(model):
            return torch.optim.Adam(model.parameters(), lr=lr)

        # def scheduler_fn(opt):
        #     if scheduler == "cosine":
        #         return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        #     if scheduler == "cosine_warmup":
        #         # Linear warmup for the first 10 % of epochs, then cosine decay
        #         warmup = max(1, epochs // 10)
        #         return torch.optim.lr_scheduler.SequentialLR(
        #             opt,
        #             schedulers=[
        #                 torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
        #                 torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=1e-6),
        #             ],
        #             milestones=[warmup],
        #         )
        #     if scheduler == "step":
        #         return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
        #     return False  # False signals training() to use no scheduler
        
        def build_scheduler_config():
            if scheduler == "cosine":
                return {
                    "name": "CosineAnnealingLR",
                    "T_max": epochs,
                    "eta_min": 1e-6,
                    "step_per_batch": False,
                }

            if scheduler == "cosine_warmup":
                warmup = max(1, epochs // 10)
                return {
                    "name": "SequentialLR",
                    "warmup_type": "linear",
                    "warmup_iters": warmup,
                    "start_factor": 1e-3,
                    "end_factor": 1.0,
                    "after_scheduler": "CosineAnnealingLR",
                    "T_max": epochs - warmup,
                    "eta_min": 1e-6,
                    "step_per_batch": False,
                }

            if scheduler == "step":
                return {
                    "name": "StepLR",
                    "step_size": max(1, epochs // 5),
                    "gamma": 0.5,
                    "step_per_batch": False,
                }

            return {
                "name": None,
                "step_per_batch": False,
            }

        cfg = {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "optimizer_fn": optimizer_fn,
            "scheduler_fn": build_scheduler_config(),
            "scheduler_step_per_batch": False,
            "patience": 15,
            "best_metric": "f1_macro",
            "save": True,
            "log_leaderboard": True,
            "verbose": False,
            # Added for test with safety bert
            # "feature_representation": "tfidf_embed_avg",
            # "embedding_model_name": "adanish91/safetybert",
        }
        result = tf_idf_train(*encoded, train_config=cfg, requirements=requirements)
        return result["best_metric_value"]

    study = optuna.create_study(direction="maximize", study_name="tf_idf_hparam_search")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study
