"""SVM classifier wrapper for feature-based classification."""

from sklearn.svm import SVC


class SVMClassifierWrapper:
    """Wrap sklearn SVC with a shared classifier interface."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        random_state: int = 42,
        class_weight: str | dict | None = None,
        probability: bool = True,
    ):
        """Initialize the SVM classifier."""
        self.model = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            class_weight=class_weight,
            probability=probability,
        )

    def fit(self, X, y):
        """Fit the classifier on feature vectors and labels."""
        return self.model.fit(X, y)

    def predict(self, X):
        """Predict class labels from feature vectors."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities from feature vectors."""
        return self.model.predict_proba(X)

    def get_params(self) -> dict:
        """Return classifier parameters for logging."""
        return self.model.get_params()