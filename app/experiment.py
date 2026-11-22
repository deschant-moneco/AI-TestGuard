"""Orchestration réutilisable : split -> réglage (validation) -> seuil -> modèle."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn

from app import config
from app.evaluation import evaluate_model
from app.prediction import FraudModel
from app.preprocessing import split_data
from app.training import select_threshold, train_model, tune_model


@dataclass
class ExperimentResult:
    model_name: str
    params: dict
    tuning: list
    threshold: float
    model: FraudModel
    splits: tuple
    val_metrics: dict
    threshold_table: pd.DataFrame


def dataset_fingerprint(df: pd.DataFrame) -> dict:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return {
        "n_rows": int(len(df)),
        "n_fraud": int(df[config.TARGET_COLUMN].sum()),
        "fraud_rate": float(df[config.TARGET_COLUMN].mean()),
        "n_duplicates": int(df.duplicated().sum()),
        "content_sha256": h.hexdigest(),
    }


def environment_info() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.ROOT_DIR,
            check=False,
        ).stdout.strip()
    except OSError:
        sha = ""
    return {
        "code_version": sha or "non versionné",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "seed": config.RANDOM_SEED,
    }


def run_experiment(
    df: pd.DataFrame,
    model_name: str = "logreg",
    params: dict | None = None,
    seed: int = config.RANDOM_SEED,
    splits=None,
) -> ExperimentResult:
    """Pipeline complet n'utilisant QUE train et validation."""
    splits = splits or split_data(df, seed=seed)
    X_train, X_val, _X_test, y_train, y_val, _y_test = splits
    tuning = []
    if params is None:
        params, tuning = tune_model(X_train, y_train, X_val, y_val, model_name, seed)
    pipe = train_model(X_train, y_train, model_name, params, seed)
    proba_val = pipe.predict_proba(X_val)[:, 1]
    threshold, table = select_threshold(y_val, proba_val)
    val_metrics = evaluate_model(y_val, (proba_val >= threshold).astype(int), proba_val)
    model = FraudModel(pipe, threshold, {"model_name": model_name, "params": params, "seed": seed})
    return ExperimentResult(
        model_name, params, tuning, threshold, model, splits, val_metrics, table
    )
