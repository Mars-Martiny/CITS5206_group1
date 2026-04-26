"""Random Forest classifier wrapper for feature-based classification."""

from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierWrapper:
    """Wrap sklearn RandomForestClassifier with a shared classifier interface."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        class_weight: str | dict | None = None,
    ):
        """Initialize the Random Forest classifier."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
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