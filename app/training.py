"""Entraînement du pipeline, choix des hyperparamètres et du seuil.

Règle d'or : hyperparamètres et seuil sont choisis **uniquement** sur le jeu de
validation. Le jeu de test n'est jamais passé à ces fonctions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app import config
from app.preprocessing import DataValidationError, preprocess_data

MODEL_FACTORIES = {
    "logreg": lambda seed: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    ),
    "tree": lambda seed: DecisionTreeClassifier(class_weight="balanced", random_state=seed),
}

# Petites grilles explorées sur la validation (modèles « légers »).
PARAM_GRIDS = {
    "logreg": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
    "tree": [
        {"max_depth": d, "min_samples_leaf": leaf} for d in (4, 6, 8, 12) for leaf in (1, 20, 100)
    ],
}


def build_pipeline(
    model_name: str = "logreg", params: dict | None = None, seed: int = config.RANDOM_SEED
) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Modèle inconnu : {model_name!r} ({list(MODEL_FACTORIES)})")
    clf = MODEL_FACTORIES[model_name](seed)
    if params:
        clf.set_params(**params)
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(preprocess_data, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y,
    model_name: str = "logreg",
    params: dict | None = None,
    seed: int = config.RANDOM_SEED,
) -> Pipeline:
    """Ajuste le pipeline complet (transformations + modèle) sur (X, y)."""
    y = np.asarray(y)
    if len(X) == 0:
        raise DataValidationError("Impossible d'entraîner sur un jeu vide")
    if len(X) != len(y):
        raise DataValidationError(f"Dimensions incohérentes : X={len(X)}, y={len(y)}")
    if set(np.unique(y)) != {0, 1}:
        raise DataValidationError("L'entraînement exige les deux classes 0 et 1")
    pipe = build_pipeline(model_name, params, seed)
    pipe.fit(X, y)
    return pipe


def tune_model(
    X_train, y_train, X_val, y_val, model_name: str, seed: int = config.RANDOM_SEED
) -> tuple[dict, list[dict]]:
    """Sélectionne les hyperparamètres maximisant la PR-AUC de validation."""
    results = []
    for params in PARAM_GRIDS[model_name]:
        pipe = train_model(X_train, y_train, model_name, params, seed)
        ap = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
        results.append({"params": params, "val_pr_auc": float(ap)})
    best = max(results, key=lambda r: r["val_pr_auc"])
    return best["params"], results


def expected_cost(
    y_true,
    y_pred,
    c_fn: float = config.COST_FALSE_NEGATIVE,
    c_fp: float = config.COST_FALSE_POSITIVE,
) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fn * c_fn + fp * c_fp


def select_threshold(
    y_val,
    proba_val,
    c_fn: float = config.COST_FALSE_NEGATIVE,
    c_fp: float = config.COST_FALSE_POSITIVE,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Seuil minimisant le coût métier attendu sur la **validation**.

    En cas d'égalité de coût, le seuil le plus élevé (moins d'alertes) est retenu.
    """
    y_val = np.asarray(y_val)
    proba_val = np.asarray(proba_val)
    if grid is None:
        grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    rows = []
    for t in grid:
        pred = (proba_val >= t).astype(int)
        tp = int(((y_val == 1) & (pred == 1)).sum())
        fp = int(((y_val == 0) & (pred == 1)).sum())
        fn = int(((y_val == 1) & (pred == 0)).sum())
        rows.append(
            {
                "threshold": float(t),
                "cost": fn * c_fn + fp * c_fp,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "fp": fp,
                "fn": fn,
            }
        )
    table = pd.DataFrame(rows)
    best_cost = table["cost"].min()
    best = table[table["cost"] == best_cost]["threshold"].max()
    return float(best), table


def refit_like(pipe: Pipeline, X, y) -> Pipeline:
    """Réajuste un clone non entraîné (utile aux tests de reproductibilité)."""
    return clone(pipe).fit(X, np.asarray(y))
