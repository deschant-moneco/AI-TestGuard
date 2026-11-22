"""Activité 6.1 (complément) — validation hors période (out-of-time).

Le split stratifié aléatoire mélange les deux journées du jeu. En production,
le modèle score des transactions FUTURES : on rejoue donc le protocole avec un
découpage chronologique (60 % / 20 % / 20 % selon Time), hyperparamètres et
seuil toujours fixés sur la validation.
"""

import json

from app import config
from app.evaluation import evaluate_model
from app.experiment import run_experiment
from app.preprocessing import load_dataset


def main():
    df = load_dataset(config.FULL_DATASET).drop_duplicates().sort_values("Time")
    n = len(df)
    tr, va, te = (
        df.iloc[: int(0.6 * n)],
        df.iloc[int(0.6 * n) : int(0.8 * n)],
        df.iloc[int(0.8 * n) :],
    )
    F, T = config.FEATURE_COLUMNS, config.TARGET_COLUMN
    splits = (tr[F], va[F], te[F], tr[T], va[T], te[T])
    out = {
        "periods_h": {
            k: [float(d.Time.min() / 3600), float(d.Time.max() / 3600)]
            for k, d in (("train", tr), ("validation", va), ("test", te))
        },
        "fraud": {k: int(d[T].sum()) for k, d in (("train", tr), ("validation", va), ("test", te))},
        "models": {},
    }
    for model_name in ("logreg", "tree"):
        exp = run_experiment(df, model_name, splits=splits)
        p = exp.model.predict_proba(te[F])
        test = evaluate_model(te[T], (p >= exp.threshold).astype(int), p)
        out["models"][model_name] = {
            "params": exp.params,
            "threshold": exp.threshold,
            "val": exp.val_metrics,
            "test": test,
        }
        print(
            model_name,
            exp.params,
            exp.threshold,
            {k: round(test[k], 4) for k in ("precision", "recall", "fpr", "pr_auc", "roc_auc")},
            test["tp"],
            test["fp"],
            test["fn"],
        )
    print(out["periods_h"], out["fraud"])
    (config.REPORTS_DIR / "temporal_validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
