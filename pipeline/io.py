import pandas as pd

def load_data(path, mode="predict"):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print("[WARN] utf-8 failed, trying latin-1...")
        df = pd.read_csv(path, encoding="latin-1")

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
