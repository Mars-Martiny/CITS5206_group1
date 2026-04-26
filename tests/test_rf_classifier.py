"""Tests for RandomForestClassifierWrapper."""

import numpy as np

from modules.models import RandomForestClassifierWrapper


def test_rf_classifier_fit_predict_and_predict_proba():
    """Test that RF classifier can fit, predict, and return probabilities."""
    X = np.random.rand(12, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    classifier = RandomForestClassifierWrapper(n_estimators=10, random_state=42)
    classifier.fit(X, y)

    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)

    assert preds.shape == (12,)
    assert probs.shape == (12, 2)


def test_rf_classifier_get_params():
    """Test that RF classifier exposes parameters for logging."""
    classifier = RandomForestClassifierWrapper(
        n_estimators=10,
        random_state=42,
        class_weight="balanced",
    )

    params = classifier.get_params()

    assert params["n_estimators"] == 10
    assert params["random_state"] == 42
    assert params["class_weight"] == "balanced"