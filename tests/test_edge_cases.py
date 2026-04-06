import pandas as pd
import pytest

from pipeline.io import load_data
from pipeline.preprocessing import preprocess
from pipeline.retrain import retrain
from prediction_models.dummy_model import DummyModel


def test_load_data_empty_csv_raises(tmp_path):
    """
    Test that loading an empty CSV raises a ValueError.
    """
    csv_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_file, index=False)

    with pytest.raises(ValueError):
        load_data(csv_file, mode="predict")


def test_load_data_missing_train_column_raises(tmp_path):
    """
    Test that missing required training columns raises a ValueError.
    """
    csv_file = tmp_path / "bad_train.csv"
    df = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
    })
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Missing required column: Energy Type"):
        load_data(csv_file, mode="train")


def test_load_data_drops_rows_with_missing_required_values(tmp_path):
    """
    Test that rows with missing required values are dropped.
    """
    csv_file = tmp_path / "missing_values.csv"
    df = pd.DataFrame({
        "Reference": ["R1", None],
        "Date and Time of Event": ["2025-01-01 10:00:00", "2025-01-02 10:00:00"],
        "Event Description": ["Worker slipped", "Worker fell"],
    })
    df.to_csv(csv_file, index=False)

    loaded = load_data(csv_file, mode="predict")
    assert len(loaded) == 1


def test_preprocess_adds_text_column():
    """
    Test that preprocess() creates the expected text column.
    """
    df = pd.DataFrame({
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
    })

    out = preprocess(df)
    assert "text" in out.columns
    assert out.loc[0, "text"] == "worker slipped mechanical"


def test_retrain_missing_label_column_raises(tmp_path):
    """
    Test that retrain() fails when the current implementation's expected
    label column is not present.
    """
    original_file = tmp_path / "original.csv"
    reviewed_file = tmp_path / "reviewed.csv"

    df_original = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
    })

    df_reviewed = pd.DataFrame({
        "Reference": ["R2"],
        "Date and Time of Event": ["2025-01-02 11:00:00"],
        "Event Description": ["Worker fell"],
        "Energy Type": ["Gravity"],
    })

    df_original.to_csv(original_file, index=False)
    df_reviewed.to_csv(reviewed_file, index=False)

    model = DummyModel()

    with pytest.raises(KeyError):
        retrain(original_file, reviewed_file, model)