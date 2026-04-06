import pandas as pd
from pipeline.output_formatter import format_output


def test_format_output_mapping():
    """
    Test that format_output() correctly maps input columns to output schema.

    This test verifies:
    - Source columns are mapped to the correct output field names
    - Key output fields exist in the result
    - Data values are preserved correctly during transformation
    """
    df = pd.DataFrame({
        "Reference": [1],
        "Event Description": ["Incident"],
        "Date and Time of Event": ["2026-04-01"],
        "predicted_energy_type": ["Energy"],
        "energy_confidence": [0.9],
    })

    result = format_output(df)

    assert "inx_id" in result.columns
    assert "incident_date" in result.columns
    assert result.loc[0, "incident_date"] == "2026-04-01"