"""Perturbations réalistes et mesure de sensibilité (Activité 5).

Justification des amplitudes :
* ``V1..V28`` : bruit gaussien proportionnel à l'écart-type de CHAQUE
  composante (mesuré sur l'entraînement), eps ∈ {1 %, 5 %, 10 %} — erreurs
  d'arrondi / de capteur plausibles ; on ne leur donne pas de sens métier ;
* ``Amount`` : variation relative ±5 % (frais, change) ; valeurs extrêmes
  ×100 et plafond du contrat (1 M€) ;
* ``Time`` : décalage +1 h (fuseau, horloge) ;
* dérive : translation de 0,5 écart-type sur V1..V28 + montants ×3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from app.evaluation import evaluate_model
from app.preprocessing import DataValidationError


def gaussian_noise_pca(X: pd.DataFrame, eps: float, stds: pd.Series, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    cols = config.PCA_FEATURES
    Xp[cols] = X[cols] + rng.normal(0, 1, (len(X), len(cols))) * (eps * stds[cols].to_numpy())
    return Xp


def scale_amount(X: pd.DataFrame, factor: float):
    return X.assign(Amount=X["Amount"] * factor)


def relative_amount_noise(X: pd.DataFrame, rel: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    return X.assign(Amount=X["Amount"] * (1 + rng.uniform(-rel, rel, len(X))))


def shift_time(X: pd.DataFrame, seconds: float):
    return X.assign(Time=X["Time"] + seconds)


def inject_missing(X: pd.DataFrame, frac_rows: float, columns, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    mask = rng.random(len(X)) < frac_rows
    Xp.loc[mask, columns] = np.nan
    return Xp, int(mask.sum())


def covariate_shift(X: pd.DataFrame, stds: pd.Series, k: float = 0.5, amount_factor=3.0):
    Xp = X.copy()
    Xp[config.PCA_FEATURES] = X[config.PCA_FEATURES] + k * stds[config.PCA_FEATURES]
    Xp["Amount"] = X["Amount"] * amount_factor
    return Xp


def sensitivity(model, X_ref, X_pert, y) -> dict:
    """Mesure l'effet d'une perturbation : probas, changements de classe, métriques."""
    p0 = model.predict_proba(X_ref)
    c0 = (p0 >= model.threshold).astype(int)
    try:
        p1 = model.predict_proba(X_pert)
    except DataValidationError as exc:
        return {"rejected": True, "reason": str(exc), "n_invalid": int(len(X_pert))}
    c1 = (p1 >= model.threshold).astype(int)
    m0, m1 = evaluate_model(y, c0, p0), evaluate_model(y, c1, p1)
    delta = np.abs(p1 - p0)
    return {
        "rejected": False,
        "mean_abs_delta_proba": float(delta.mean()),
        "max_abs_delta_proba": float(delta.max()),
        "n_class_changes": int((c0 != c1).sum()),
        "pct_class_changes": float((c0 != c1).mean()),
        "delta_recall": m1["recall"] - m0["recall"],
        "delta_precision": m1["precision"] - m0["precision"],
        "delta_fpr": m1["fpr"] - m0["fpr"],
        "recall": m1["recall"],
        "precision": m1["precision"],
        "fpr": m1["fpr"],
        "alert_rate": float(c1.mean()),
        "n_invalid": 0,
    }
