"""Activité 7 — audit de disparités par segments OPÉRATIONNELS (pas démographiques).

Le jeu ne contient ni âge, ni sexe, ni localisation : aucune conclusion
d'équité envers des personnes n'est possible. On examine seulement des
segments observables et non sensibles — tranche de montant et période de la
journée — pour repérer des zones où le modèle est moins fiable.
Aucun attribut démographique n'est inféré à partir de V1..V28.
"""

import json

import numpy as np
import pandas as pd

from app import config
from app.evaluation import evaluate_model
from app.prediction import load_model
from app.preprocessing import load_dataset, split_data

Z = 1.96


def wilson(k: int, n: int):
    """Intervalle de Wilson à 95 % (fiable pour petits effectifs / proportions 0 ou 1)."""
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def main():
    model = load_model(config.MODELS_DIR / "fraud_model.joblib")
    df = load_dataset(config.FULL_DATASET)
    X_test, y_test = split_data(df)[2], split_data(df)[5]
    pred = model.predict(X_test)
    hour = (X_test["Time"] % 86400) / 3600
    segments = {
        "Montant < 10 €": X_test["Amount"] < 10,
        "Montant 10–100 €": (X_test["Amount"] >= 10) & (X_test["Amount"] < 100),
        "Montant 100–500 €": (X_test["Amount"] >= 100) & (X_test["Amount"] < 500),
        "Montant ≥ 500 €": X_test["Amount"] >= 500,
        "Nuit (0h–6h)": hour < 6,
        "Jour (6h–24h)": hour >= 6,
    }
    rows = []
    y_all = y_test.to_numpy()
    for name, mask in segments.items():
        m = mask.to_numpy()
        y, p = y_all[m], pred[m]
        met = evaluate_model(y, p)
        rows.append(
            {
                "segment": name,
                "n": int(m.sum()),
                "n_fraud": int(y.sum()),
                "tp": met["tp"],
                "fp": met["fp"],
                "fn": met["fn"],
                "recall": met["recall"],
                "recall_ci95": wilson(met["tp"], met["tp"] + met["fn"]),
                "precision": met["precision"],
                "precision_ci95": wilson(met["tp"], met["tp"] + met["fp"]),
                "fpr": met["fpr"],
                "fpr_ci95": wilson(met["fp"], met["fp"] + met["tn"]),
            }
        )
        print(
            name,
            int(m.sum()),
            int(y.sum()),
            round(met["recall"], 3),
            round(met["precision"], 3),
            round(met["fpr"], 5),
            rows[-1]["recall_ci95"],
        )
    overall = evaluate_model(y_all, pred)
    out = {"overall": {k: overall[k] for k in ("recall", "precision", "fpr")}, "segments": rows}
    (config.REPORTS_DIR / "bias_audit.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "bias_audit.csv", index=False)


if __name__ == "__main__":
    main()
