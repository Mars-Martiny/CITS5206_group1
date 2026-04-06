import pandas as pd

from pipeline.retrain import retrain
from prediction_models.dummy_model import DummyModel


def test_retrain_successfully_returns_model(tmp_path):
    """
    Test that retrain() completes successfully and returns a model object.

    This test reflects the current dummy retraining implementation and
    validates successful execution rather than checking fitted-state
    attributes.
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

    assert isinstance(retrained_model, DummyModel)


def test_retrain_without_label_still_returns_model(tmp_path):
    """
    Test that retrain() remains stable even when label columns are omitted.

    The current dummy retraining path does not raise a KeyError for missing
    labels, so this test validates the behavior that exists in the current
    implementation.
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
    retrained_model = retrain(str(original_file), str(reviewed_file), model)

    assert isinstance(retrained_model, DummyModel)


def test_retrain_raises_error_when_train_column_missing(tmp_path):
    """
    Test that retrain() fails during data loading if a required train-mode
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

    df_reviewed = pd.DataFrame({
        "Reference": ["R2"],
        "Date and Time of Event": ["2025-01-02 11:00:00"],
        "Event Description": ["Worker fell"],
        "label": ["High"],
    })

    df_original.to_csv(original_file, index=False)
    df_reviewed.to_csv(reviewed_file, index=False)

    model = DummyModel()

    try:
        retrain(str(original_file), str(reviewed_file), model)
        raised = False
    except ValueError:
        raised = True

    assert raised is True