import pandas as pd
import pytest

from pipeline.retrain import retrain
from prediction_models.dummy_model import DummyModel


def test_retrain_successfully_fits_model(tmp_path):
    """
    Test that retrain() loads both datasets, preprocesses them,
    and fits the provided model successfully.
    """
    original_file = tmp_path / "original.csv"
    reviewed_file = tmp_path / "reviewed.csv"

    df_original = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
        "label": ["Energy"],
    })

    df_reviewed = pd.DataFrame({
        "Reference": ["R2"],
        "Date and Time of Event": ["2025-01-02 11:00:00"],
        "Event Description": ["Worker fell from ladder"],
        "Energy Type": ["Gravity"],
        "label": ["High"],
    })

    df_original.to_csv(original_file, index=False)
    df_reviewed.to_csv(reviewed_file, index=False)

    model = DummyModel()
    retrained_model = retrain(str(original_file), str(reviewed_file), model)

    assert retrained_model.is_fitted is True


def test_retrain_raises_error_when_label_missing(tmp_path):
    """
    Test that retrain() raises an error if the expected label column
    is not present in the merged training data.
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
        retrain(str(original_file), str(reviewed_file), model)


def test_retrain_raises_error_when_train_column_missing(tmp_path):
    """
    Test that retrain() fails during loading if a required training
    column is missing from either input file.
    """
    original_file = tmp_path / "original.csv"
    reviewed_file = tmp_path / "reviewed.csv"

    df_original = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
        "label": ["Energy"],
    })

    # Missing Energy Type
    df_reviewed = pd.DataFrame({
        "Reference": ["R2"],
        "Date and Time of Event": ["2025-01-02 11:00:00"],
        "Event Description": ["Worker fell"],
        "label": ["High"],
    })

    df_original.to_csv(original_file, index=False)
    df_reviewed.to_csv(reviewed_file, index=False)

    model = DummyModel()

    with pytest.raises(ValueError, match="Missing required column: Energy Type"):
        retrain(str(original_file), str(reviewed_file), model)
        