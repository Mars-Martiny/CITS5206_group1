"""Run full feature-based inference through existing pipeline."""

from modules.inference import run_inference
from modules.models.feature_classifier_wrapper import FeatureClassifierWrapper


def run_feature_pipeline(
    feature_extractor,
    classifier,
    train_dl,
    test_dl,
    target: str = "energy",
    device: str = "cpu",
):
    """Train classifier on extracted features, then run inference + evaluation."""

    # extract features from training set and train classifier
    X_train, y_train_energy, y_train_risk = [], [], []

    for D, DL, Energy, Risk in train_dl:
        if DL is not None:
            features = feature_extractor(D, DL)
        else:
            features = feature_extractor(D)

        X_train.append(features.detach().cpu())
        y_train_energy.append(Energy)
        y_train_risk.append(Risk)

    import torch

    X_train = torch.cat(X_train).numpy()
    y_train_energy = torch.cat(y_train_energy).numpy()
    y_train_risk = torch.cat(y_train_risk).numpy()

    y_train = y_train_energy if target == "energy" else y_train_risk

    # train RF/SVM
    classifier.fit(X_train, y_train)

    # wrapped into PyTorch model
    model = FeatureClassifierWrapper(feature_extractor, classifier)

    # build config for inference
    config = {
        "model": model,
        "device": device,
        "need_length": True,
        "energy_model": target == "energy",
        "test_dl": test_dl,
    }

    return run_inference(config)