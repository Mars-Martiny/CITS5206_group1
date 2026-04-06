import pandas as pd

from pipeline.io import load_data
from pipeline.preprocessing import preprocess


def retrain(original_path, reviewed_path, model):
    """
    Retrain a model using original and reviewed labelled datasets.

    The function loads both datasets in training mode, concatenates them,
    applies preprocessing, and fits the provided model on the combined data.

    Args:
        original_path (str): Path to the original labelled CSV file.
        reviewed_path (str): Path to the reviewed labelled CSV file.
        model: A model object that provides a ``fit()`` method.

    Returns:
        The retrained model instance.

    Notes:
        This implementation assumes that the combined dataset contains a
        ``"label"`` column after loading and preprocessing.
    """
    df_original = load_data(original_path, mode="train")
    df_reviewed = load_data(reviewed_path, mode="train")

    df_all = pd.concat([df_original, df_reviewed], ignore_index=True)

    print(f"[INFO] Retraining on {len(df_all)} samples")

    df_all = preprocess(df_all)

    X = df_all["text"]
    y = df_all["label"]

    model.fit(X, y)

    return model