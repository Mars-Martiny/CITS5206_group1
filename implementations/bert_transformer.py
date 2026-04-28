"""Run BERT transformer experiments for Energy Type and Risk classification."""

import torch
import torch.optim as optim

from modules.data_loader.bert_loader import df_to_bert_dataloader
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_tokenizer import BertTokenizerWrapper
from modules.embedding.bert_embedding import BertEmbeddingBackend
from modules.encoding.label_encoder import LabelEncoder
from modules.models.bert_classifier import BertClassifier
from modules.training_loop import training


def encode_label_column(train_df, valid_df, test_df, label_col):
    """Fit label encoder on train labels and transform train/valid/test.

    :param train_df: Training dataframe containing the label column.
    :type train_df: pandas.DataFrame
    :param valid_df: Validation dataframe containing the label column.
    :type valid_df: pandas.DataFrame
    :param test_df: Test dataframe containing the label column.
    :type test_df: pandas.DataFrame
    :param label_col: Name of the column containing label data.
    :type label_col: str

    :returns: Tuple of encoded dataframes, label encoder, and class names.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, LabelEncoder, list[str]]
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[label_col].astype(str).tolist())

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df[label_col] = label_encoder.encode_many(
        train_df[label_col].astype(str).tolist()
    )
    valid_df[label_col] = label_encoder.encode_many(
        valid_df[label_col].astype(str).tolist()
    )
    test_df[label_col] = label_encoder.encode_many(
        test_df[label_col].astype(str).tolist()
    )

    class_names = [
        label_encoder.id_to_label[i]
        for i in range(label_encoder.num_classes)
    ]

    return train_df, valid_df, test_df, label_encoder, class_names


def run_bert_experiment(
    train_df,
    valid_df,
    test_df,
    text_col,
    label_col,
    run_name,
    fine_tune=False,
    pooling="mean",
    batch_size=8,
    epochs=5,
    learning_rate=None,
    dropout=0.2,
    max_length=160,
    use_class_weights=False,
    weight_decay=0.01,
    threshold=0.8,
):
    """Train and evaluate a BERT classifier for one target label.

    :param train_df: Training dataframe.
    :type train_df: pandas.DataFrame
    :param valid_df: Validation dataframe.
    :type valid_df: pandas.DataFrame
    :param test_df: Test dataframe.
    :type test_df: pandas.DataFrame
    :param text_col: Name of the column containing text data.
    :type text_col: str
    :param label_col: Name of the column containing label data.
    :type label_col: str
    :param run_name: Name for this training run.
    :type run_name: str
    :param fine_tune: Whether to fine-tune BERT or keep it frozen.
    :type fine_tune: bool
    :param pooling: Pooling strategy for sentence embedding, either "cls" or "mean".
    :type pooling: str
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :param epochs: Maximum number of training epochs.
    :type epochs: int
    :param learning_rate: Learning rate. If None, uses 2e-5 when fine-tuning and 1e-4 when frozen.
    :type learning_rate: float | None
    :param dropout: Dropout rate for classifier head.
    :type dropout: float
    :param max_length: Maximum sequence length for BERT tokenizer.
    :type max_length: int
    :param use_class_weights: Whether to use class weights in the loss function.
    :type use_class_weights: bool
    :param weight_decay: Weight decay factor for AdamW.
    :type weight_decay: float
    :param threshold: Confidence threshold for auto-classification analysis.
    :type threshold: float

    :returns: Run summary from the shared training pipeline.
    :rtype: dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, valid_df, test_df, label_encoder, class_names = encode_label_column(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_col=label_col,
    )

    bert_config = BertEmbeddingConfig(
        model_name="bert-base-uncased",
        max_length=max_length,
        dropout=0.1,
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
        num_classes=label_encoder.num_classes,
        dropout=dropout,
    ).to(device)

    if learning_rate is None:
        learning_rate = 2e-5 if fine_tune else 1e-4

    optimiser = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion_weights = None

    if use_class_weights:
        class_counts = train_df[label_col].value_counts().sort_index()
        weights = 1.0 / class_counts
        weights = weights / weights.mean()
        criterion_weights = torch.tensor(weights.values, dtype=torch.float)

    run_summary = training(
        model=model,
        energy_model=True,
        model_type="BERT",
        need_length=False,

        optimiser=optimiser,
        optimiser_args=None,

        scheduler=None,
        scheduler_step_per_batch=False,

        criterion_type="cross_entropy",
        criterion_weights=criterion_weights,
        criterion_args=None,

        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,

        epochs=epochs,
        patience=2,
        num_classes=label_encoder.num_classes,
        class_dict=label_encoder.id_to_label,
        clip_grad_max_norm=1.0,

        best_metric="f1_macro",
        best_metric_mode=None,

        threshold=threshold,
        temperature=1.0,
        use_temperature=False,

        parameters={
            "model_name": "bert-base-uncased",
            "pooling": pooling,
            "fine_tune": fine_tune,
            "batch_size": batch_size,
            "label_col": label_col,
            "text_col": text_col,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "max_length": max_length,
            "use_class_weights": use_class_weights,
            "weight_decay": weight_decay,
        },
        device=device,

        compute_train_metrics=False,
        save=True,
        parent_dir="trained_models",
        run_name=run_name,

        extra_config={
            "class_names": class_names,
            "label_col": label_col,
            "text_col": text_col,
            "pooling": pooling,
            "fine_tune": fine_tune,
        },
    )

    return run_summary