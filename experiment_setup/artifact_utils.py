"""Utility functions for working with artifact pickle files created by the training and retraining runners.

artifact_path_from_result:
    Get the path to the artifacts pickle file from a training result dictionary.

update_artifact_pickle:
    Merge updates into the just-saved artifacts pickle file.
"""

import pickle
from typing import Any
from pathlib import Path
from datetime import datetime


# Helpers for updating runner artifact pickle files.
def artifact_path_from_result(result: dict[str, Any]) -> Path:
    """Get the path to the artifacts pickle file from a training result dictionary.
    
    Args:
        result: The training result dictionary returned by the train function.
    
    Returns:
        The Path to the artifacts pickle file associated with the training result.
    """
    cfg = result["config"]
    return Path(cfg["save_dir"]) / f"{cfg['save_name']}_artifacts.pkl"


def update_artifact_pickle(result: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge updates into the just-saved artifacts pickle.
    
    Args:
        result: The training result dictionary returned by the train function.
        updates: A dict of updates to merge into the artifacts pickle.
    
    Returns:
        None. The artifacts pickle file is updated in-place.
    """
    path = _artifact_path_from_result(result)

    with open(path, "rb") as f:
        artifacts = pickle.load(f)

    artifacts.update(updates)
    artifacts.setdefault("retrain_history", [])
    artifacts["retrain_history"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **updates.get("retrain_metadata", {}),
        }
    )

    with open(path, "wb") as f:
        pickle.dump(artifacts, f)