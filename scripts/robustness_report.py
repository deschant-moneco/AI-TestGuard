"""Activité 5 — tableau des scénarios de robustesse sur le jeu complet.

Usage : python scripts/robustness_report.py (après run_experiment.py)
"""

import json

import numpy as np

from app import config
from app.monitoring import alert_rate_check, drift_report
from app.prediction import load_model
from app.preprocessing import load_dataset, split_data
from app.robustness import (
    covariate_shift,
    gaussian_noise_pca,
    inject_missing,
    relative_amount_noise,
    scale_amount,
    sensitivity,
    shift_time,
)


def main():
    model = load_model(config.MODELS_DIR / "fraud_model.joblib")
    df = load_dataset(config.FULL_DATASET)
    X_train, X_val, X_test, _, _, y_test = split_data(df)
    stds = X_train.std()
    ref_rate = float(model.predict(X_val).mean())

    def missing(cols, frac):
        return inject_missing(X_test, frac, cols, seed=1)[0]

    def inf_vals():
        Xp = X_test.copy()
        idx = np.random.default_rng(2).random(len(Xp)) < 0.05
        Xp.loc[idx, "V14"] = np.inf
        return Xp

    def non_numeric():
        Xp = X_test.astype({"V10": object})
        Xp.iloc[0, Xp.columns.get_loc("V10")] = "N/A"
        return Xp

    scenarios = [
        (
            "1a",
            "Manquants : 5 % des lignes, V4",
            "traiter (imputation médiane)",
            lambda: missing(["V4"], 0.05),
        ),
        (
            "1b",
            "Manquants : 10 % des lignes, V4+V14+Amount",
            "traiter (imputation médiane)",
            lambda: missing(["V4", "V14", "Amount"], 0.10),
        ),
        (
            "1c",
            "Manquants : 5 % des lignes, toutes variables",
            "traiter + alerte qualité",
            lambda: missing(config.FEATURE_COLUMNS, 0.05),
        ),
        ("2a", "Infinis : 5 % des lignes sur V14", "traiter (inf→NaN→médiane)", inf_vals),
        ("2b", "Valeur non numérique dans V10", "rejeter", non_numeric),
        (
            "3a",
            "Amount ×100 (tous)",
            "traiter (≤ 1 M€)",
            lambda: X_test.assign(Amount=np.minimum(X_test["Amount"] * 100, 1e6)),
        ),
        (
            "3b",
            "Amount = 1 000 000 € (plafond)",
            "traiter + alerte métier",
            lambda: X_test.assign(Amount=1e6),
        ),
        ("3c", "Amount ×10⁷ (au-delà du contrat)", "rejeter", lambda: scale_amount(X_test, 1e7)),
        (
            "4a",
            "Bruit gaussien V1–V28, ε=1 % σ",
            "traiter (sensibilité mesurée)",
            lambda: gaussian_noise_pca(X_test, 0.01, stds),
        ),
        (
            "4b",
            "Bruit gaussien V1–V28, ε=5 % σ",
            "traiter (sensibilité mesurée)",
            lambda: gaussian_noise_pca(X_test, 0.05, stds),
        ),
        (
            "4c",
            "Bruit gaussien V1–V28, ε=10 % σ",
            "traiter (sensibilité mesurée)",
            lambda: gaussian_noise_pca(X_test, 0.10, stds),
        ),
        ("4d", "Amount ±5 % (uniforme)", "traiter", lambda: relative_amount_noise(X_test, 0.05)),
        ("4e", "Time +1 h", "traiter", lambda: shift_time(X_test, 3600)),
        (
            "5a",
            "Dérive : V +0,5σ et Amount ×3",
            "alerte (dérive)",
            lambda: covariate_shift(X_test, stds, 0.5, 3.0),
        ),
        (
            "5b",
            "Dérive : V −0,5σ",
            "alerte (dérive)",
            lambda: covariate_shift(X_test, stds, -0.5, 1.0),
        ),
        ("6a", "Schéma : colonne V1 absente", "rejeter", lambda: X_test.drop(columns=["V1"])),
        (
            "6b",
            "Schéma : colonne Amount renommée",
            "rejeter",
            lambda: X_test.rename(columns={"Amount": "amount"}),
        ),
    ]
    rows = []
    for sid, desc, expected, make in scenarios:
        Xp = make()
        res = sensitivity(model, X_test, Xp, y_test)
        if not res["rejected"] and sid.startswith("5"):
            rep = drift_report(X_train, Xp, config.FEATURE_COLUMNS)
            res["n_psi_alerts"] = int((rep["status"] == "alerte").sum())
            res["rate_alert"] = alert_rate_check(model.predict(Xp), ref_rate)["alert"]
        rows.append({"id": sid, "scenario": desc, "expected": expected, **res})
        print(sid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    base = sensitivity(model, X_test, X_test, y_test)
    out = {
        "reference": {
            "recall": base["recall"],
            "precision": base["precision"],
            "fpr": base["fpr"],
            "alert_rate": base["alert_rate"],
            "val_alert_rate": ref_rate,
        },
        "scenarios": rows,
    }
    (config.REPORTS_DIR / "robustness.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
