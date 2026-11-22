"""Jeux de données et modèles partagés par les tests.

* ``synthetic_df`` : petit jeu synthétique conforme au contrat (tests unitaires,
  rapides, sans dépendance au fichier Kaggle) ;
* ``sample_df`` : échantillon réel versionné ``data/sample_creditcard.csv`` ;
* ``trained`` : expérience complète (split, entraînement, seuil) sur l'échantillon.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from app import config
from app.experiment import run_experiment
from app.preprocessing import load_dataset

BASELINE_PARAMS = {"C": 0.01}


def make_synthetic(n: int = 400, fraud_rate: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    if n >= 2:
        y[:2] = [0, 1]  # garantit la présence des deux classes
    data = {"Time": np.sort(rng.uniform(0, 172_800, n))}
    for i, col in enumerate(config.PCA_FEATURES):
        shift = 2.0 if i < 5 else 0.0  # V1..V5 séparent les classes
        data[col] = rng.normal(0, 1, n) + shift * y
    data["Amount"] = rng.lognormal(3, 1.2, n)
    data["Class"] = y
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return load_dataset(config.SAMPLE_DATASET)


@pytest.fixture(scope="session")
def trained(sample_df):
    return run_experiment(sample_df, "logreg", params=BASELINE_PARAMS)


@pytest.fixture(scope="session")
def model(trained):
    return trained.model


@pytest.fixture(scope="session")
def X_test(trained):  # noqa: N802 (convention ML)
    return trained.splits[2]


@pytest.fixture(scope="session")
def y_test(trained):
    return trained.splits[5]


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    caplog.set_level(logging.INFO)
