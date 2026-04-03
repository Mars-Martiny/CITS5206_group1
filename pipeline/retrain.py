import pandas as pd
from pipeline.io import load_data
from pipeline.preprocessing import preprocess

def retrain(original_path, reviewed_path, model):
    df_original = load_data(original_path, mode="train")
    df_reviewed = load_data(reviewed_path, mode="train")

    df_all = pd.concat([df_original, df_reviewed], ignore_index=True)

    print(f"[INFO] Retraining on {len(df_all)} samples")

    df_all = preprocess(df_all)

    X = df_all["text"]
    y = df_all["Energy Type"]

    model.fit(X, y)

    return model
