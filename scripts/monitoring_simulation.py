"""Activité 10 — alertes simulées : référence (train) vs flux « production »."""

import json

from app import config
from app.monitoring import alert_rate_check, drift_report
from app.prediction import load_model
from app.preprocessing import load_dataset, split_data
from app.robustness import covariate_shift


def main():
    model = load_model(config.MODELS_DIR / "fraud_model.joblib")
    df = load_dataset(config.FULL_DATASET)
    X_train, X_val, X_test, *_ = split_data(df)
    ref_rate = float(model.predict(X_val).mean())
    stds = X_train.std()
    # Flux simulés : J1 = test (même loi), J2 = dérive modérée, J3 = 2e jour
    # chronologique seul (Time > 24 h), J4 = dérive forte inverse.
    streams = {
        "J1 — test (même distribution)": X_test,
        "J2 — dérive V +0,5σ / Amount ×3": covariate_shift(X_test, stds, 0.5, 3.0),
        "J3 — seconde journée uniquement": X_test[X_test["Time"] > 86_400],
        "J4 — dérive V −0,5σ": covariate_shift(X_test, stds, -0.5, 1.0),
    }
    out = {"reference_alert_rate": ref_rate, "streams": {}}
    for name, X in streams.items():
        rep = drift_report(X_train, X, config.FEATURE_COLUMNS)
        rate = alert_rate_check(model.predict(X), ref_rate)
        out["streams"][name] = {
            "n": int(len(X)),
            "psi_alerts": int((rep.status == "alerte").sum()),
            "psi_warnings": int((rep.status == "attention").sum()),
            "top_psi": rep.head(5).to_dict("records"),
            "alert_rate": rate,
        }
        print(
            name,
            out["streams"][name]["psi_alerts"],
            out["streams"][name]["psi_warnings"],
            rep.head(3).to_dict("records"),
            round(rate["rate"], 5),
            rate["alert"],
        )
    (config.REPORTS_DIR / "monitoring_simulation.json").write_text(
        json.dumps(out, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
