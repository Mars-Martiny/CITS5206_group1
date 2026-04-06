import pandas as pd


def load_data(path, mode="predict"):
    """
    Load input data from a CSV file and validate required columns.

    The function reads the CSV file into a pandas DataFrame, checks whether
    the file is empty, validates the required columns based on the selected
    mode, and drops rows with missing values in the required fields.

    Args:
        path (str): Path to the input CSV file.
        mode (str, optional): Data loading mode. Supported values are
            ``"train"`` and ``"predict"``. Defaults to ``"predict"``.

    Returns:
        pd.DataFrame: A validated DataFrame with rows containing missing
        required values removed.

    Raises:
        ValueError: If the CSV file is empty.
        ValueError: If the mode is invalid.
        ValueError: If any required column is missing.
    """
    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError("Input CSV is empty")

    print(f"[INFO] Loaded {len(df)} rows from {path}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    if mode == "train":
        required_columns = ["Reference", "Date and Time of Event", "Event Description", "Energy Type"]
    elif mode == "predict":
        required_columns = ["Reference", "Date and Time of Event", "Event Description"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    before = len(df)
    df = df.dropna(subset=required_columns)
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows due to missing values")

    return df