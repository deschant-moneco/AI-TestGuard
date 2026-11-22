"""Calcul des métriques adaptées à une classe positive rare."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app import config


def _check_inputs(y_true, y_pred, y_proba):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    y_proba = None if y_proba is None else np.asarray(y_proba, dtype=float)
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true doit être un vecteur non vide")
    if len(y_pred) != len(y_true) or (y_proba is not None and len(y_proba) != len(y_true)):
        raise ValueError("y_true, y_pred et y_proba doivent avoir la même longueur")
    if not set(np.unique(y_true)) <= {0, 1} or not set(np.unique(y_pred)) <= {0, 1}:
        raise ValueError("Les étiquettes doivent valoir 0 ou 1")
    if y_proba is not None and (
        not np.isfinite(y_proba).all() or (y_proba < 0).any() or (y_proba > 1).any()
    ):
        raise ValueError("Les probabilités doivent être finies et dans [0, 1]")
    return y_true, y_pred, y_proba


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """Retourne précision, rappel, F1, ROC-AUC, PR-AUC, FPR, FNR et la matrice.

    Les AUC valent ``None`` si une seule classe est présente (non définies).
    """
    y_true, y_pred, y_proba = _check_inputs(y_true, y_pred, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    metrics = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_proba is not None and both_classes:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return metrics


def check_acceptance(metrics: dict, criteria: dict | None = None) -> dict:
    """Compare les métriques aux critères d'acceptation fixés a priori."""
    c = criteria or config.ACCEPTANCE_CRITERIA
    checks = {
        "recall": (metrics["recall"], ">=", c["recall_min"]),
        "precision": (metrics["precision"], ">=", c["precision_min"]),
        "fpr": (metrics["fpr"], "<=", c["fpr_max"]),
        "pr_auc": (metrics["pr_auc"], ">=", c["pr_auc_min"]),
        "roc_auc": (metrics["roc_auc"], ">=", c["roc_auc_min"]),
    }
    out = {}
    for name, (value, op, limit) in checks.items():
        ok = value is not None and (value >= limit if op == ">=" else value <= limit)
        out[name] = {"value": value, "rule": f"{op} {limit}", "passed": bool(ok)}
    out["accepted"] = all(v["passed"] for v in out.values())
    return out


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str,
    n_boot: int = 1000,
    seed: int = config.RANDOM_SEED,
    alpha: float = 0.05,
):
    """Intervalle de confiance bootstrap (percentile) d'une métrique à seuil."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = evaluate_model(y_true[idx], y_pred[idx])
        values.append(m[metric])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
