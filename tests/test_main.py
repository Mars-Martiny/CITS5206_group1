import sys
import pandas as pd

import main


def test_main_train_mode_runs(monkeypatch, tmp_path):
    """
    Test that main() dispatches correctly in train mode.

    This test isolates the main entry-point behavior and avoids depending
    on the full underlying pipeline implementation by monkeypatching the
    imported functions used by main.py.
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

    monkeypatch.setattr(main, "load_data", lambda *args, **kwargs: pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Mechanical"],
        "predicted_energy_type": ["Energy"],
    }))
    monkeypatch.setattr(main, "preprocess", lambda df: df.assign(text=["worker slipped mechanical"]))

    main.main()


def test_main_predict_mode_runs(monkeypatch, tmp_path):
    """
    Test that main() dispatches correctly in predict mode.

    The test monkeypatches the imported pipeline functions so that the
    main entry point can be validated without relying on the full
    formatter/inference contract of the current dummy implementation.
    """
    csv_file = tmp_path / "predict.csv"
    df = pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
    })
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "predict", "--input", str(csv_file)],
    )

    monkeypatch.setattr(main, "load_data", lambda *args, **kwargs: pd.DataFrame({
        "Reference": ["R1"],
        "Date and Time of Event": ["2025-01-01 10:00:00"],
        "Event Description": ["Worker slipped"],
        "Energy Type": ["Unknown"],
    }))
    monkeypatch.setattr(main, "preprocess", lambda df: df.assign(text=["worker slipped unknown"]))
    monkeypatch.setattr(main, "predict", lambda model, df: df.assign(
        predicted_energy_type=["Energy"],
        energy_confidence=[0.9],
        energy_score=[0.9],
        predicted_damage_potential=["High"],
        damage_confidence=[0.8],
        damage_score=[0.8],
        fatal_flag=[1],
        energy_action_required=[False],
        damage_action_required=[False],
        inx_id=["R1"],
        incident_date=["2025-01-01 10:00:00"],
        incident_description=["Worker slipped"],
    ))

    main.main()


def test_main_retrain_mode_runs(monkeypatch, tmp_path):
    """
    Test that main() dispatches correctly in retrain mode.

    This validates that the retrain branch of the command-line entry point
    executes without errors when the retrain dependency is satisfied.
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

    monkeypatch.setattr(main, "retrain", lambda *args, **kwargs: object())

    main.main()