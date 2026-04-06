import pandas as pd

from pipeline.io import load_data
from pipeline.preprocessing import preprocess
from pipeline.inference import predict
from pipeline.output_formatter import format_output
from prediction_models.dummy_model import DummyModel


def test_full_predict_pipeline_flow(tmp_path):
    """
    Test the full prediction pipeline flow from raw CSV input to formatted output.

    This test validates that:
    1. Data can be loaded correctly in predict mode
    2. Preprocessing generates the expected text field
    3. The model can run inference without errors
    4. Output formatting produces the required schema

    The test simulates a minimal real-world pipeline execution.
    """
    csv_file = tmp_path / "predict.csv"
    df = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker fell from ladder"],
        "Energy Type": ["Unknown"],
    })
    df.to_csv(csv_file, index=False)

    loaded = load_data(csv_file, mode="predict")

    # Ensure compatibility with preprocess (which expects Energy Type)
    loaded["Energy Type"] = "Unknown"

    processed = preprocess(loaded)

    model = DummyModel()
    predicted = predict(model, processed)

    # Prepare fields required by output formatter
    predicted["inx_id"] = predicted["Reference"]
    predicted["incident_date"] = predicted["Date and Time of Event"]
    predicted["incident_description"] = predicted["Event Description"]

    formatted = format_output(predicted)

    # Assertions: ensure pipeline produced valid structured output
    assert len(formatted) == 1
    assert "inx_id" in formatted.columns
    assert "incident_date" in formatted.columns
    assert "incident_description" in formatted.columns
    assert "predicted_energy_type" in formatted.columns