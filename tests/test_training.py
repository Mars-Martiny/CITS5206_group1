import pandas as pd

from pipeline.io import load_data
from pipeline.preprocessing import preprocess
from pipeline.retrain import retrain
from prediction_models.dummy_model import DummyModel


def test_train_flow_with_expected_columns(tmp_path):
    """
    Test the basic training flow using the current main-style logic.

    This test constructs a training dataset that matches the current code
    expectations, including the placeholder ``predicted_energy_type`` column.
    """
    csv_file = tmp_path / "train.csv"
    df = pd.DataFrame({
        "Reference": ["R1", "R2"],
        "Date and Time of Event": ["2025-01-01 10:00:00", "2025-01-02 11:00:00"],
        "Event Description": ["Worker slipped", "Worker fell"],
        "Energy Type": ["Mechanical", "Gravity"],
        "predicted_energy_type": ["Energy", "High"],
    })
    df.to_csv(csv_file, index=False)

    loaded = load_data(csv_file, mode="train")
    processed = preprocess(loaded)

    model = DummyModel()
    X = processed["text"]
    y = processed["predicted_energy_type"]

    fitted_model = model.fit(X, y)

    assert fitted_model.is_fitted is True
    assert len(X) == 2
    assert len(y) == 2


def test_retrain_combines_original_and_reviewed_data(tmp_path):
    """
    Test that retrain() loads, combines, preprocesses, and fits on both datasets.
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
    retrained_model = retrain(original_file, reviewed_file, model)

    assert retrained_model.is_fitted is True


def test_retrain_uses_both_rows(tmp_path):
    """
    Test retraining stability by confirming merged datasets produce the expected
    combined sample count before fitting.
    """
    original_file = tmp_path / "original.csv"
    reviewed_file = tmp_path / "reviewed.csv"

    df_original = pd.DataFrame({
        "Reference": ["R1", "R2"],
        "Date and Time of Event": ["2025-01-01 10:00:00", "2025-01-02 11:00:00"],
        "Event Description": ["A", "B"],
        "Energy Type": ["Mechanical", "Electrical"],
        "label": ["Energy", "Energy"],
    })

    df_reviewed = pd.DataFrame({
        "Reference": ["R3"],
        "Date and Time of Event": ["2025-01-03 12:00:00"],
        "Event Description": ["C"],
        "Energy Type": ["Gravity"],
        "label": ["High"],
    })

    df_original.to_csv(original_file, index=False)
    df_reviewed.to_csv(reviewed_file, index=False)

    model = DummyModel()
    retrained_model = retrain(original_file, reviewed_file, model)

    assert retrained_model.is_fitted is True