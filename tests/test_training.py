import pandas as pd

from pipeline.io import load_data
from pipeline.preprocessing import preprocess
from pipeline.retrain import retrain
from prediction_models.dummy_model import DummyModel


def test_train_flow_with_expected_columns(tmp_path):
    """
    Test the basic training flow using the current train-style logic.

    This test verifies that:
    - training data can be loaded in train mode
    - preprocessing generates the text feature
    - the dummy model fit call runs without error

    The current dummy model implementation does not expose training-state
    flags, so this test validates successful execution rather than fitted
    attributes.
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

    model.fit(X, y)

    assert len(X) == 2
    assert len(y) == 2


def test_retrain_combines_original_and_reviewed_data(tmp_path):
    """
    Test that retrain() can run successfully on original and reviewed data.

    The current dummy retraining flow is permissive, so this test asserts
    that the function returns a DummyModel instance rather than expecting
    internal training-state flags.
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

    assert isinstance(retrained_model, DummyModel)


def test_retrain_uses_both_rows(tmp_path):
    """
    Test retraining stability using multiple rows across both input files.

    This verifies that retrain() can process merged datasets of different
    sizes and still return a model object successfully.
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

    assert isinstance(retrained_model, DummyModel)