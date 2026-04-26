"""Tests for SVMClassifierWrapper."""

import numpy as np

from modules.models import SVMClassifierWrapper


def test_svm_classifier_fit_predict_and_predict_proba():
    """Test that SVM classifier can fit, predict, and return probabilities."""
    X = np.random.rand(12, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    classifier = SVMClassifierWrapper(kernel="linear", C=1.0, random_state=42)
    classifier.fit(X, y)

    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)

    assert preds.shape == (12,)
    assert probs.shape == (12, 2)


def test_svm_classifier_get_params():
    """Test that SVM classifier exposes parameters for logging."""
    classifier = SVMClassifierWrapper(
        kernel="linear",
        C=0.5,
        random_state=42,
        class_weight="balanced",
    )

    params = classifier.get_params()

    assert params["kernel"] == "linear"
    assert params["C"] == 0.5
    assert params["random_state"] == 42
    assert params["class_weight"] == "balanced"