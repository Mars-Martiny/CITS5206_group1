"""Run full feature-based inference through the existing project leaderboard."""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from modules.models.feature_classifier_wrapper import FeatureClassifierWrapper


def _extract_features(feature_extractor, dataloader, device: str = "cpu"):
    """Extract feature vectors and labels from a sequence dataloader."""
    X, y_energy, y_risk = [], [], []

    feature_extractor.to(device)
    feature_extractor.eval()

    with torch.no_grad():
        for D, DL, Energy, Risk in dataloader:
            D = D.to(device)

            if DL is not None:
                DL = DL.to(device)
                features = feature_extractor(D, DL)
            else:
                features = feature_extractor(D)

            X.append(features.detach().cpu())
            y_energy.append(Energy.detach().cpu())
            y_risk.append(Risk.detach().cpu())

    return (
        torch.cat(X).numpy(),
        torch.cat(y_energy).numpy(),
        torch.cat(y_risk).numpy(),
    )


def _build_metrics(y_true, y_pred, class_dict):
    """Build metric dictionary compatible with leaderboard logger."""
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    return {
        "loss": None,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "class_metrics": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "auto_classification_rate": None,
        "fatal_flag_count": None,
        "fatal_flag_rate": None,
        "meets_requirement": None,
        "threshold_used": None,
        "confidence_high_rate": None,
        "confidence_medium_rate": None,
        "confidence_low_rate": None,
        "req_high_confidence_met": None,
        "fatal_accuracy": None,
        "req_fatal_accuracy_met": None,
        "per_class_requirements": None,
        "req_all_f1_targets_met": None,
    }


def run_feature_pipeline(
    feature_extractor,
    classifier,
    train_dl,
    test_dl,
    target: str = "energy",
    device: str = "cpu",
    model_type: str = "SVM_BiLSTM_Features",
    run_name: str | None = None,
    parent_dir: str = "trained_models",
    leaderboard_dir: str = "leaderboard",
    class_dict: dict | None = None,
    log_leaderboard: bool = True,
):
    """Train classical classifier on extracted features, evaluate, and log run."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"{model_type.lower()}_{timestamp}"
    class_dict = class_dict or {}

    X_train, y_train_energy, y_train_risk = _extract_features(
        feature_extractor, train_dl, device
    )
    X_test, y_test_energy, y_test_risk = _extract_features(
        feature_extractor, test_dl, device
    )

    y_train = y_train_energy if target == "energy" else y_train_risk
    y_test = y_test_energy if target == "energy" else y_test_risk

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_test)
    else:
        probabilities = None

    model = FeatureClassifierWrapper(feature_extractor, classifier)
    metrics = _build_metrics(y_test, predictions, class_dict)

    save_dir = Path(parent_dir) / f"{timestamp}_{run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{run_name}_model.pkl"
    summary_path = save_dir / f"{run_name}_run_summary.json"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "feature_extractor_state_dict": feature_extractor.state_dict(),
                "classifier": classifier,
                "target": target,
                "model_type": model_type,
            },
            f,
        )

    run_summary = {
        "history": {
            "training": {
                "val": {
                    key: [value]
                    for key, value in metrics.items()
                }
            },
            "test": {
                key: [value]
                for key, value in metrics.items()
            },
        },
        "best_epoch": 1,
        "best_metric_name": "accuracy",
        "best_metric_value": metrics["accuracy"],
        "best_model_state_dict": None,
        "training_time_sec": time.time() - start_time,
    }

    config = {
        "run_name": run_name,
        "save_name": run_name,
        "model_type": model_type,
        "model": model,
        "energy_model": target == "energy",
        "epochs": 1,
        "patience": None,
        "optimiser": None,
        "scheduler": None,
        "criterion_type": None,
        "scheduler_step_per_batch": None,
        "clip_grad_max_norm": None,
        "class_dict": class_dict,
        "num_classes": len(set(y_train.tolist()) | set(y_test.tolist())),
        "metadata": {
            "model_class": model.__class__.__name__,
            "feature_extractor_class": feature_extractor.__class__.__name__,
            "classifier_class": classifier.__class__.__name__,
            "optimiser_class": None,
            "scheduler_class": None,
            "criterion_class": None,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    k: v
                    for k, v in config.items()
                    if k != "model"
                },
                "metrics": metrics,
                "model_path": str(model_path),
                "predictions": predictions.tolist(),
                "targets": y_test.tolist(),
            },
            f,
            indent=2,
            default=str,
        )

    if log_leaderboard:
        from modules.leaderboard.logger import log_run

        log_run(
            run_summary=run_summary,
            config=config,
            model_path=str(model_path),
            leaderboard_dir=leaderboard_dir,
        )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "targets": y_test,
        "probabilities": probabilities,
        "model_path": str(model_path),
        "summary_path": str(summary_path),
    }