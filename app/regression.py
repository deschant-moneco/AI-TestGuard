"""Comparaison d'une version candidate à la baseline figée."""

from __future__ import annotations

from app import config

HIGHER_IS_BETTER = {
    "recall": True,
    "precision": True,
    "pr_auc": True,
    "roc_auc": True,
    "fpr": False,
}


def compare_to_baseline(
    baseline_metrics: dict, candidate_metrics: dict, tolerance: dict | None = None
) -> dict:
    """Retourne, par métrique, l'écart et un statut ; plus une décision globale.

    * ``reject``      : au moins une dégradation dépasse la tolérance ;
    * ``investigate`` : aucune dégradation hors tolérance mais un écart
      (dans un sens ou l'autre) > tolérance/2 — à expliquer avant d'accepter ;
    * ``accept``      : écarts négligeables ou améliorations.
    """
    tol = tolerance or config.REGRESSION_TOLERANCE
    rows, decision = {}, "accept"
    for metric, higher in HIGHER_IS_BETTER.items():
        base, cand = baseline_metrics[metric], candidate_metrics[metric]
        delta = cand - base
        degradation = -delta if higher else delta
        if degradation > tol[metric]:
            status = "régression"
            decision = "reject"
        elif abs(delta) > tol[metric] / 2:
            status = "écart à expliquer" if degradation > 0 else "amélioration"
            if decision == "accept" and degradation > 0:
                decision = "investigate"
        else:
            status = "stable"
        rows[metric] = {
            "baseline": base,
            "candidate": cand,
            "delta": delta,
            "tolerance": tol[metric],
            "status": status,
        }
    return {"metrics": rows, "decision": decision}
