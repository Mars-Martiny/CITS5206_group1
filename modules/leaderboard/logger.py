"""Leaderboard CSV logger.

Appends one row per training run to leaderboard/leaderboard.csv.
New columns from future runs are merged in; old rows keep NaN for any
columns that didn't exist when they were written.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from .runner import resolve_runner


def log_run(run_summary: dict, config: dict, model_path: str, leaderboard_dir: str = "leaderboard") -> None:
    """Appends one row per training run to leaderboard/leaderboard.csv.
    
    New columns from future runs are merged in; old rows keep NaN for any
    columns that didn't exist when they were written.

    Args:
        run_summary: Dictionary of run summary constructed from train_loop, contains 
            config, history, best_epoch, best_metric_name, best_metric_value, best_model_state_dict, training_time_sec
        config: Dictionary containing model, dataloaders, optimizer, scheduler,
            criterion, epochs, device, patience, and all other training parameters.
        model_path: String for path of the saved model
        leaderboard_dir: String where the leaderboard is solved, defaults to "leaderboard"
    """
    leaderboard_dir = Path(leaderboard_dir)
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = leaderboard_dir / "leaderboard.csv"

    row = _build_row(run_summary, config, model_path, leaderboard_dir)

    if leaderboard_path.exists():
        existing = pd.read_csv(leaderboard_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(leaderboard_path, index=False)
    print(f"Leaderboard updated: {leaderboard_path}")


def _build_row(run_summary: dict, config: dict, model_path: str, leaderboard_dir: Path) -> dict:
    """Build the singular entry row.

    Args:
        run_summary: Dictionary of run summary constructed from train_loop, contains 
            config, history, best_epoch, best_metric_name, best_metric_value, best_model_state_dict, training_time_sec
        config: Dictionary containing model, dataloaders, optimizer, scheduler,
            criterion, epochs, device, patience, and all other training parameters.
        model_path: String for path of the saved model
        leaderboard_dir: String where the leaderboard is solved, defaults to "leaderboard"
    """
    history = run_summary.get("history", {})
    best_epoch = run_summary.get("best_epoch") or 1
    val_history = history.get("training", {}).get("val", {})
    test_history = history.get("test", {})
    idx = best_epoch - 1  # history is 0-indexed

    def _at_best(metric):
        vals = val_history.get(metric, [])
        return vals[idx] if idx < len(vals) else None

    def _last_test(metric):
        vals = test_history.get(metric, [])
        return vals[-1] if vals else None

    metadata = config.get("metadata", {})

    # lr lives on the live optimiser object; fall back gracefully if absent
    optimiser = config.get("optimiser")
    try:
        lr = optimiser.defaults.get("lr") if optimiser is not None else None
    except Exception:
        lr = None

    return {
        # --- Identity ---
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": config.get("run_name") or config.get("save_name"),
        "runner": resolve_runner(leaderboard_dir),
        "git_commit": _get_git_commit_hash(),
        # --- Location ---
        "model_path": str(model_path),
        # --- Best checkpoint ---
        "best_epoch": best_epoch,
        "best_metric_name": run_summary.get("best_metric_name"),
        "best_metric_value": run_summary.get("best_metric_value"),
        # --- Val metrics at best epoch ---
        "val_loss": _at_best("loss"),
        "val_accuracy": _at_best("accuracy"),
        "val_f1_macro": _at_best("f1_macro"),
        "val_f1_weighted": _at_best("f1_weighted"),
        "val_precision_macro": _at_best("precision_macro"),
        "val_recall_macro": _at_best("recall_macro"),
        # --- Hyperparameters ---
        "epochs_max": config.get("epochs"),
        "patience": config.get("patience"),
        "lr": lr,
        "clip_grad_max_norm": config.get("clip_grad_max_norm"),
        "criterion_type": config.get("criterion_type"),
        "scheduler_step_per_batch": config.get("scheduler_step_per_batch"),
        # --- Pipeline specifics ---
        "model_type": config.get("model_type"),
        "model_class": metadata.get("model_class"),
        "optimiser_class": metadata.get("optimiser_class"),
        "scheduler_class": metadata.get("scheduler_class"),
        "criterion_class": metadata.get("criterion_class"),
        "energy_model": config.get("energy_model"),
        # --- Runtime ---
        "training_time_sec": run_summary.get("training_time_sec"),
        # --- Test set metrics ---
        "test_loss": _last_test("loss"),
        "test_accuracy": _last_test("accuracy"),
        "test_f1_macro": _last_test("f1_macro"),
        "test_f1_weighted": _last_test("f1_weighted"),
        "test_precision_macro": _last_test("precision_macro"),
        "test_recall_macro": _last_test("recall_macro"),
        "test_auto_classification_rate": _last_test("auto_classification_rate"),
        "test_fatal_flag_rate": _last_test("fatal_flag_rate"),
        # --- Client requirement results ---
        "req_high_confidence_met": _last_test("req_high_confidence_met"),
        "confidence_high_rate": _last_test("confidence_high_rate"),
        "confidence_medium_rate": _last_test("confidence_medium_rate"),
        "confidence_low_rate": _last_test("confidence_low_rate"),
        "test_fatal_accuracy": _last_test("fatal_accuracy"),
        "req_fatal_accuracy_met": _last_test("req_fatal_accuracy_met"),
        "req_all_f1_targets_met": _last_test("req_all_f1_targets_met"),
        "per_class_requirements": json.dumps(_last_test("per_class_requirements")),
    }


def _get_git_commit_hash() -> str:
    """Get the commit hash.
    
    Returns:
        String for the short hash of the current HEAD commit, or empty string if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""
