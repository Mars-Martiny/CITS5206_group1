def predict(model, df):
    texts = df["text"].tolist()

    predictions = model.predict(texts)
    confidences = model.predict_proba(texts).max(axis=1)

    df["predicted_energy_type"] = predictions
    df["energy_confidence"] = confidences

    return df
