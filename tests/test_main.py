import sys
import pandas as pd

import main


def test_main_train_mode_runs(monkeypatch, tmp_path):
    """
    Test that main() dispatches correctly in train mode.
    """
    csv_file = tmp_path / "train.csv"
    df = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
        "predicted_energy_type": ["Energy"],
    })
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "train", "--input", str(csv_file)],
    )

    main.main()


def test_main_predict_mode_runs(monkeypatch, tmp_path):
    """
    Test that main() dispatches correctly in predict mode with a compatible
    placeholder input.
    """
    csv_file = tmp_path / "predict.csv"
    df = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Unknown"],
    })
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "predict", "--input", str(csv_file)],
    )

    main.main()


def test_main_retrain_mode_runs(monkeypatch, tmp_path):
    """
    Test that main() dispatches correctly in retrain mode.
    """
    original_file = tmp_path / "original.csv"
    reviewed_file = tmp_path / "reviewed.csv"

    df_original = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
        "label": ["Energy"],
    })

    df_reviewed = pd.DataFrame({
        "Reference": ["R2"],
        "Date and Time of Event": ["2025-01-02 11:00:00"],
        "Event Description": ["Worker fell"],
        "Energy Type": ["Gravity"],
        "label": ["High"],
    })

    df_original.to_csv(original_file, index=False)
    df_reviewed.to_csv(reviewed_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "retrain",
            "--input",
            str(original_file),
            "--reviewed",
            str(reviewed_file),
        ],
    )

    main.main()