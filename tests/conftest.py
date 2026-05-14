"""Shared fixtures for integration tests.

Fast fixtures (TF-IDF, BiGRU) run by default.
Slow fixtures (BERT, LoopedTransformer) require: pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal synthetic dataset – raw column names matching actual incident CSVs
# ---------------------------------------------------------------------------

_ROWS = [
    ("R001", "01/01/2023", "Worker fell from scaffold at height", "Gravitational", "Fatal"),
    ("R002", "02/01/2023", "Vehicle collision near site entrance", "Vehicular", "Life Altering"),
    ("R003", "03/01/2023", "Electrical fault caused short circuit", "Electrical", "Temporary / Minor Damage"),
    ("R004", "04/01/2023", "Heavy object dropped from crane height", "Gravitational", "Life Altering"),
    ("R005", "05/01/2023", "Forklift struck worker in warehouse area", "Vehicular", "Fatal"),
    ("R006", "06/01/2023", "Power line contact during maintenance work", "Electrical", "Fatal"),
    ("R007", "07/01/2023", "Slip on wet surface near electrical panel", "Gravitational", "Temporary / Minor Damage"),
    ("R008", "08/01/2023", "Truck reversed into loading bay barrier", "Vehicular", "Temporary / Minor Damage"),
    ("R009", "09/01/2023", "Cable failure caused equipment drop event", "Electrical", "Life Altering"),
    ("R010", "10/01/2023", "Worker tripped on scaffold walkway surface", "Gravitational", "Temporary / Minor Damage"),
    ("R011", "11/01/2023", "Car ran stop sign at site intersection", "Vehicular", "Life Altering"),
    ("R012", "12/01/2023", "Generator overload caused fire hazard", "Electrical", "Fatal"),
    ("R013", "13/01/2023", "Ladder fell during roof access attempt", "Gravitational", "Life Altering"),
    ("R014", "14/01/2023", "Mining truck incident at road junction", "Vehicular", "Temporary / Minor Damage"),
    ("R015", "15/01/2023", "Circuit breaker fault during routine inspection", "Electrical", "Temporary / Minor Damage"),
    ("R016", "16/01/2023", "Fall from elevated work platform structure", "Gravitational", "Fatal"),
    ("R017", "17/01/2023", "Forklift tipped while carrying heavy load", "Vehicular", "Fatal"),
    ("R018", "18/01/2023", "Electrical arc flash during switchboard work", "Electrical", "Life Altering"),
    ("R019", "19/01/2023", "Worker slipped on gravel near drop zone", "Gravitational", "Temporary / Minor Damage"),
    ("R020", "20/01/2023", "Light vehicle struck berm at high speed", "Vehicular", "Life Altering"),
    ("R021", "21/01/2023", "Substation fault during peak demand period", "Electrical", "Fatal"),
    ("R022", "22/01/2023", "Rigging failure during heavy crane lift", "Gravitational", "Life Altering"),
    ("R023", "23/01/2023", "Road train near miss at blind corner area", "Vehicular", "Fatal"),
    ("R024", "24/01/2023", "Switchgear arcing during scheduled maintenance", "Electrical", "Temporary / Minor Damage"),
    ("R025", "25/01/2023", "Person fell into open excavation trench", "Gravitational", "Life Altering"),
    ("R026", "26/01/2023", "Bus driver lost control on steep descent", "Vehicular", "Temporary / Minor Damage"),
    ("R027", "27/01/2023", "Underground cable struck by excavator bucket", "Electrical", "Fatal"),
    ("R028", "28/01/2023", "Object dropped from scaffold upper work level", "Gravitational", "Fatal"),
    ("R029", "29/01/2023", "Truck door opened into path of cyclist", "Vehicular", "Life Altering"),
    ("R030", "30/01/2023", "Transformer oil leak ignited near building", "Electrical", "Temporary / Minor Damage"),
]

_COLUMNS = [
    "Reference",
    "Date and Time of Event",
    "Detailed Description of Event",
    "Energy Type",
    "Type of Potential Damage",
]


def _make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(_ROWS, columns=_COLUMNS)


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    """Three split CSV files with raw incident column names."""
    base = tmp_path_factory.mktemp("splits")
    df = _make_raw_df()
    splits = {
        "train": (base / "train.csv", df.iloc[:20]),
        "valid": (base / "valid.csv", df.iloc[20:25]),
        "test":  (base / "test.csv",  df.iloc[25:]),
    }
    paths: dict[str, Path] = {}
    for name, (path, frame) in splits.items():
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# Session-scoped model fixtures (train once, reuse across all tests)
# ---------------------------------------------------------------------------

def _train(raw_csv, model_type: str, architecture: str, extra_cfg: dict | None = None):
    from api import train
    cfg = {"epochs": 2, **(extra_cfg or {})}
    result = train(
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        model_type=model_type,
        architecture=architecture,
        train_config=cfg,
    )
    return result, result["config"]["save_dir"]


@pytest.fixture(scope="session")
def tfidf_energy_model(raw_csv):
    return _train(raw_csv, "energy", "tf_idf")


@pytest.fixture(scope="session")
def tfidf_damage_model(raw_csv):
    return _train(raw_csv, "damage", "tf_idf")


@pytest.fixture(scope="session")
def bigru_energy_model(raw_csv):
    return _train(raw_csv, "energy", "bigru")


@pytest.fixture(scope="session")
def bert_energy_model(raw_csv):
    return _train(raw_csv, "energy", "bert", {"epochs": 1})


@pytest.fixture(scope="session")
def looped_energy_model(raw_csv):
    return _train(raw_csv, "energy", "looped_transformer", {"epochs": 1})
