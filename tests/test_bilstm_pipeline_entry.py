import pandas as pd
import torch

from implementations.bilstm import (
    _normalise_tokens,
    prepare_bilstm_dataloaders,
    run_bilstm_training,
)
from modules.models import BiLSTMClassifier


def _sample_df():
    return pd.DataFrame(
        {
            "tokens": [
                ["worker", "fell"],
                ["machine", "jammed"],
                "['vehicle', 'moved']",
                "chemical spill",
            ],
            "energy_type": ["Human", "Mechanical", "Vehicular", "Chemical"],
        }
    )


def test_normalise_tokens_accepts_lists_and_csv_strings():
    assert _normalise_tokens(["a", "b"]) == ["a", "b"]
    assert _normalise_tokens("['a', 'b']") == ["a", "b"]
    assert _normalise_tokens("a b") == ["a", "b"]


def test_prepare_bilstm_dataloaders_produces_sequence_batches():
    df = _sample_df()

    train_dl, valid_dl, test_dl, metadata = prepare_bilstm_dataloaders(
        df,
        df,
        df,
        tokens_col="tokens",
        label_col="energy_type",
        batch_size=2,
        max_length=5,
    )

    D, DL, Energy, Risk = next(iter(train_dl))

    assert D.shape[0] == 2
    assert D.shape[1] <= 5
    assert DL.shape == (2,)
    assert DL.max().item() <= 5
    assert Energy.shape == (2,)
    assert Risk.shape == (2,)
    assert metadata["vocab_encoder"].vocab_size >= 2
    assert metadata["label_encoder"].num_classes == 4
    assert valid_dl.batch_size == 2
    assert test_dl.batch_size == 2


def test_run_bilstm_training_passes_bilstm_to_shared_training(monkeypatch):
    captured = {}

    def fake_training(**kwargs):
        captured.update(kwargs)
        return {"config": kwargs}

    monkeypatch.setattr("implementations.bilstm.training", fake_training)

    result = run_bilstm_training(
        _sample_df(),
        _sample_df(),
        _sample_df(),
        tokens_col="tokens",
        label_col="energy_type",
        batch_size=2,
        max_length=5,
        epochs=1,
        save=False,
        log_leaderboard=False,
    )

    assert isinstance(captured["model"], BiLSTMClassifier)
    assert captured["model_type"] == "BiLSTM"
    assert captured["need_length"] is True
    assert captured["energy_model"] is True

    D, DL, _, _ = next(iter(captured["train_dl"]))
    with torch.no_grad():
        logits = captured["model"](D, DL)

    assert logits.shape == (2, 4)
    assert result["config"] == captured
