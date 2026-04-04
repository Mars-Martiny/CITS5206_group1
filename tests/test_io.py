import pandas as pd
import pytest
from pipeline.io import load_data


def test_load_data_predict_success(tmp_path):
    file_path = tmp_path / "predict.csv"
    df = pd.DataFrame({
        "Reference": [1],
        "Date and Time of Event": ["2026-04-01"],
        "Event Description": ["Event A"],
    })
    df.to_csv(file_path, index=False)

    result = load_data(file_path, mode="predict")

    assert len(result) == 1


def test_load_data_missing_column(tmp_path):
    file_path = tmp_path / "bad.csv"
    df = pd.DataFrame({
        "Reference": [1],
    })
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_data(file_path, mode="predict")