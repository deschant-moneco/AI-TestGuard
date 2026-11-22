"""Activité 6.2 — compare des versions candidates à la baseline FIGÉE.

La comparaison se fait sur le jeu de VALIDATION fixe (même split, même graine).
Usage : python scripts/compare_versions.py
"""

import json

from app import config
from app.experiment import run_experiment
from app.preprocessing import load_dataset, split_data
from app.regression import compare_to_baseline

CANDIDATES = {
    "v1.1-logreg-C1": ("logreg", {"C": 1.0}, "Régularisation plus faible (C=1)"),
    "v1.2-tree-d12": (
        "tree",
        {"max_depth": 12, "min_samples_leaf": 20},
        "Remplacement par l'arbre de décision réglé",
    ),
    "v1.3-logreg-nobalance": (
        "logreg",
        {"C": 0.01, "class_weight": None},
        "Suppression de la pondération des classes",
    ),
}


def main():
    baseline = json.loads((config.BASELINE_DIR / "baseline.json").read_text())
    snap = baseline["full"]
    df = load_dataset(config.FULL_DATASET)
    splits = split_data(df)
    out = {
        "baseline": {
            "model": baseline["model"],
            "params": baseline["params"],
            "threshold": snap["threshold"],
            "code_version": baseline["environment"]["code_version"],
            "metrics": snap["validation_metrics"],
        },
        "candidates": {},
    }
    # Contrôle : la baseline rejouée doit être « accept »
    replay = run_experiment(df, baseline["model"], params=baseline["params"], splits=splits)
    out["replay_baseline"] = compare_to_baseline(
        snap["validation_metrics"], replay.val_metrics, baseline["tolerance"]
    )
    for name, (model_name, params, desc) in CANDIDATES.items():
        exp = run_experiment(df, model_name, params=params, splits=splits)
        cmp = compare_to_baseline(
            snap["validation_metrics"], exp.val_metrics, baseline["tolerance"]
        )
        # Garde-fou : un seuil en bordure de grille signale des probabilités
        # mal calibrées / un optimum non atteint -> au minimum « investigate ».
        if exp.threshold in (0.01, 0.99) and cmp["decision"] == "accept":
            cmp["decision"] = "investigate"
            cmp["note"] = f"Seuil optimal en bordure de grille ({exp.threshold})"
        out["candidates"][name] = {
            "description": desc,
            "model": model_name,
            "params": params,
            "threshold": exp.threshold,
            **cmp,
        }
        print(
            name,
            cmp["decision"],
            {m: round(r["delta"], 4) for m, r in cmp["metrics"].items()},
            "thr",
            exp.threshold,
        )
    print("replay:", out["replay_baseline"]["decision"])
    (config.REPORTS_DIR / "regression_comparison.json").write_text(
        json.dumps(out, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
