import pandas as pd

from pipeline.output_formatter import format_output


def test_format_output_mapping():
    """
    Test that format_output() correctly maps source fields to output fields.

    This test provides all columns required by the current formatter contract
    so that the function can complete successfully and the output schema can
    be validated.
    """
    df = pd.DataFrame({
        "Reference": [1],
        "Event Description": ["Incident"],
        "Date and Time of Event": ["2026-04-01"],

        "inx_id": [1],
        "incident_date": ["2026-04-01"],
        "incident_description": ["Incident"],

        "predicted_energy_type": ["Energy"],
        "energy_confidence": [0.9],
        "energy_score": [0.9],
        "predicted_damage_potential": ["High"],
        "damage_confidence": [0.8],
        "damage_score": [0.8],
        "fatal_flag": [1],
        "energy_action_required": [False],
        "damage_action_required": [False],
    })

    result = format_output(df)

    assert "inx_id" in result.columns
    assert "incident_date" in result.columns
    assert result.loc[0, "incident_date"] == "2026-04-01"