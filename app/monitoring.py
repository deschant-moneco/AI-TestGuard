"""Surveillance post-déploiement : dérive des données et du taux d'alertes.

* **Data drift** : la distribution des entrées change (PSI par variable,
  calculable immédiatement, sans label).
* **Concept drift** : la relation entrées -> fraude change ; détectable
  seulement quand les labels (rétrofacturations) arrivent, via rappel / FPR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WARNING = 0.10
PSI_ALERT = 0.25


def psi(reference, current, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index avec bins par quantiles de la référence."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("Échantillons vides")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + eps
    q = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=None) -> pd.DataFrame:
    columns = columns or list(reference.columns)
    rows = []
    for c in columns:
        v = psi(reference[c], current[c])
        level = "alerte" if v >= PSI_ALERT else "attention" if v >= PSI_WARNING else "ok"
        rows.append({"feature": c, "psi": round(v, 4), "status": level})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def alert_rate_check(predictions, expected_rate: float, tolerance_factor: float = 2.0) -> dict:
    """Alerte si le taux de transactions signalées s'écarte d'un facteur > tolérance."""
    rate = float(np.mean(predictions))
    low, high = expected_rate / tolerance_factor, expected_rate * tolerance_factor
    return {
        "rate": rate,
        "expected": expected_rate,
        "low": low,
        "high": high,
        "alert": not (low <= rate <= high),
    }
