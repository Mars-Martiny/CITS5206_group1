def preprocess(df):
    df["text"] = df["Event Description"].astype(str) + " " + df["Energy Type"].astype(str)
    df["text"] = df["text"].str.lower()
    return df
