import pandas as pd
from pipeline.preprocessing import preprocess
from pipeline.inference import predict
from pipeline.output_formatter import format_output
from prediction_models.dummy_model import DummyModel


def test_pipeline_smoke():
    df = pd.DataFrame({
        "Reference": [1],
        "Date and Time of Event": ["2026-04-01"],
        "Event Description": ["Incident"],
        "Energy Type": ["Mechanical"],
    })

    model = DummyModel()

    df = preprocess(df)
    df = predict(model, df)
    df_out = format_output(df)

    assert len(df_out) == 1