import pandas as pd

OUTPUT_COLUMN_MAPPING = {
    "Reference": "inx_id",
    "Event Description": "incident_description",
    "Date and Time of Event": "incident_date"
}

MODEL_OUTPUT_COLUMNS = [
    "predicted_energy_type",
    "energy_confidence"
]

def format_output(df):
    df_out = pd.DataFrame()

    for src_col, out_col in OUTPUT_COLUMN_MAPPING.items():
        if src_col not in df.columns:
            raise ValueError(f"Missing column: {src_col}")
        df_out[out_col] = df[src_col]

    for col in MODEL_OUTPUT_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing model output column: {col}")
        df_out[col] = df[col]

    return df_out
