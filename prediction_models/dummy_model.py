import numpy as np


class DummyModel:
    """
    A simple placeholder model used for testing pipeline integration.

    This dummy implementation provides ``fit()``, ``predict()``, and
    ``predict_proba()`` methods so the rest of the pipeline can run before
    a real trained model is integrated.
    """

    def __init__(self):
        """
        Initialise the dummy model with default class labels.
        """
        self.classes_ = np.array(["Energy", "Damage"])
        self.is_fitted = False

    def fit(self, X, y):
        """
        Mark the dummy model as fitted.

        Args:
            X: Training input samples.
            y: Training labels.

        Returns:
            DummyModel: The fitted dummy model instance.
        """
        self.is_fitted = True
        return self

    def predict(self, texts):
        """
        Generate placeholder predictions from input texts.

        Args:
            texts (list[str]): A list of input text samples.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self.is_fitted:
            pass

        predictions = []
        for text in texts:
            text = str(text).lower()
            if "fall" in text or "gravity" in text:
                predictions.append("High")
            else:
                predictions.append("Energy")
        return np.array(predictions)

    def predict_proba(self, texts):
        """
        Generate placeholder class probabilities from input texts.

        Args:
            texts (list[str]): A list of input text samples.

        Returns:
            np.ndarray: Predicted class probabilities with shape
            ``(n_samples, n_classes)``.
        """
        probs = []
        for text in texts:
            text = str(text).lower()
            if "fall" in text or "gravity" in text:
                probs.append([0.2, 0.8])
            else:
                probs.append([0.8, 0.2])
        return np.array(probs)