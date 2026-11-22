"""Activité 3 + 4 + 6.1 : comparaison des modèles, évaluation finale, sauvegarde.

Usage : python scripts/run_experiment.py [--data data/creditcard.csv]

Protocole :
1. split stratifié 60/20/20 (après suppression des doublons) ;
2. pour chaque modèle : hyperparamètres + seuil choisis sur VALIDATION ;
3. choix du modèle retenu sur le coût métier de VALIDATION ;
4. évaluation sur TEST une seule fois, pour tous les modèles (aucune décision
   n'est prise à partir de ces chiffres) ;
5. sauvegarde du pipeline retenu, rechargement et contrôle de cohérence.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from app import config  # noqa: E402
from app.evaluation import bootstrap_ci, check_acceptance, evaluate_model  # noqa: E402
from app.experiment import dataset_fingerprint, environment_info, run_experiment  # noqa: E402
from app.prediction import load_model, save_model  # noqa: E402
from app.preprocessing import load_dataset, split_data  # noqa: E402
from app.training import expected_cost  # noqa: E402

NAMES = {"logreg": "Régression logistique", "tree": "Arbre de décision"}


def main(data_path):
    t0 = time.perf_counter()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    fp = dataset_fingerprint(df)
    splits = split_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # --- contrôles anti-fuite ---------------------------------------------------
    idx = [set(s.index) for s in (X_train, X_val, X_test)]
    leak = {
        "overlap_train_val": len(idx[0] & idx[1]),
        "overlap_train_test": len(idx[0] & idx[2]),
        "overlap_val_test": len(idx[1] & idx[2]),
        "duplicated_rows_train_test": int(pd.concat([X_train, X_test]).duplicated().sum()),
    }
    split_info = {
        name: {"n": int(len(y)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean())}
        for name, y in (("train", y_train), ("validation", y_val), ("test", y_test))
    }

    results, curves = {}, {}
    # Référence naïve : prédit toujours « normale » -> accuracy élevée, rappel nul
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    d_pred = dummy.predict(X_test)
    results["dummy"] = {
        "name": "Référence naïve (toujours normale)",
        "test": evaluate_model(y_test, d_pred, dummy.predict_proba(X_test)[:, 1]),
    }

    experiments = {}
    for model_name in ("logreg", "tree"):
        exp = run_experiment(df, model_name, splits=splits)
        experiments[model_name] = exp
        pipe, thr = exp.model.pipeline, exp.threshold
        p_train = pipe.predict_proba(X_train)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (p_test >= thr).astype(int)
        test_m = evaluate_model(y_test, pred_test, p_test)
        test_m_05 = evaluate_model(y_test, (p_test >= 0.5).astype(int), p_test)
        results[model_name] = {
            "name": NAMES[model_name],
            "params": exp.params,
            "tuning": exp.tuning,
            "threshold": thr,
            "val": exp.val_metrics,
            "val_cost": expected_cost(y_val, (pipe.predict_proba(X_val)[:, 1] >= thr)),
            "train": evaluate_model(y_train, (p_train >= thr).astype(int), p_train),
            "test": test_m,
            "test_at_0_5": test_m_05,
            "test_cost": expected_cost(y_test, pred_test),
            "test_recall_ci95": bootstrap_ci(y_test, pred_test, "recall"),
            "test_precision_ci95": bootstrap_ci(y_test, pred_test, "precision"),
            "acceptance_test": check_acceptance(test_m),
            "classification_report": classification_report(
                y_test,
                pred_test,
                labels=[0, 1],
                target_names=["Normale", "Fraude"],
                zero_division=0,
                digits=4,
            ),
        }
        curves[model_name] = (p_test, pred_test)
        exp.threshold_table.to_csv(
            config.REPORTS_DIR / f"threshold_table_{model_name}.csv", index=False
        )

    # Sélection du modèle retenu : coût métier sur VALIDATION (jamais le test)
    champion = min(("logreg", "tree"), key=lambda m: results[m]["val_cost"])

    # --- figures ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in ("logreg", "tree"):
        p, _ = curves[m]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{NAMES[m]} (AUC={results[m]['test']['roc_auc']:.3f})")
        axes[1].plot(rec, prec, label=f"{NAMES[m]} (AP={results[m]['test']['pr_auc']:.3f})")
        axes[1].scatter(
            results[m]["test"]["recall"],
            results[m]["test"]["precision"],
            marker="o",
            s=40,
            zorder=3,
        )
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Aléatoire")
    axes[0].set(
        xlabel="Taux de faux positifs (FPR)",
        ylabel="Rappel (TPR)",
        title="Courbe ROC — jeu de test",
    )
    axes[1].axhline(
        y_test.mean(),
        color="k",
        ls="--",
        lw=0.8,
        label=f"Aléatoire (taux fraude={y_test.mean():.4f})",
    )
    axes[1].set(
        xlabel="Rappel",
        ylabel="Précision",
        title="Courbe précision-rappel — jeu de test (• = seuil retenu)",
    )
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, m in zip(axes, ("logreg", "tree"), strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            curves[m][1],
            labels=[0, 1],
            display_labels=["Normale", "Fraude"],
            ax=ax,
            colorbar=False,
            values_format="d",
        )
        ax.set_title(f"{NAMES[m]} (seuil={results[m]['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ("logreg", "tree"):
        tab = experiments[m].threshold_table
        ax.plot(tab["threshold"], tab["cost"], label=NAMES[m])
        ax.axvline(results[m]["threshold"], ls=":", color="grey")
    ax.set(
        xlabel="Seuil de décision",
        ylabel="Coût attendu (€) sur validation",
        title=f"Choix du seuil sur VALIDATION (FN={config.COST_FALSE_NEGATIVE:.0f}€, "
        f"FP={config.COST_FALSE_POSITIVE:.0f}€)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "threshold_cost_validation.png", dpi=130)
    plt.close(fig)

    # --- sauvegarde / rechargement du modèle retenu ---------------------------
    exp = experiments[champion]
    exp.model.metadata.update({"dataset": fp, "environment": environment_info()})
    path = save_model(exp.model, config.MODELS_DIR / "fraud_model.joblib")
    reloaded = load_model(path)
    diff = float(
        np.max(
            np.abs(reloaded.predict_proba(X_test) - exp.model.pipeline.predict_proba(X_test)[:, 1])
        )
    )
    same_labels = bool((reloaded.predict(X_test) == curves[champion][1]).all())

    out = {
        "dataset": fp,
        "environment": environment_info(),
        "splits": split_info,
        "leakage_checks": leak,
        "costs": {"fn": config.COST_FALSE_NEGATIVE, "fp": config.COST_FALSE_POSITIVE},
        "acceptance_criteria": config.ACCEPTANCE_CRITERIA,
        "results": results,
        "champion": champion,
        "reload_check": {
            "max_abs_proba_diff": diff,
            "identical_labels": same_labels,
            "model_path": str(path.relative_to(config.ROOT_DIR)),
            "model_size_kb": round(path.stat().st_size / 1024, 1),
        },
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    write_markdown(out)
    print(
        json.dumps(
            {
                m: {
                    k: results[m]["test"][k]
                    for k in (
                        "precision",
                        "recall",
                        "f1",
                        "fpr",
                        "fnr",
                        "roc_auc",
                        "pr_auc",
                        "accuracy",
                        "tp",
                        "fp",
                        "fn",
                        "tn",
                    )
                }
                for m in results
            },
            indent=1,
        )
    )
    print(
        "champion:",
        champion,
        "thresholds:",
        {m: results[m]["threshold"] for m in ("logreg", "tree")},
        "params:",
        {m: results[m]["params"] for m in ("logreg", "tree")},
    )
    print("reload:", out["reload_check"], "leak:", leak, "durée:", out["duration_s"])


def _fmt(v, pct=False):
    if v is None:
        return "n/d"
    return f"{100 * v:.2f} %" if pct else f"{v:.4f}"


def write_markdown(out):
    r = out["results"]
    lines = [
        "# Comparaison des modèles — jeu de test (généré automatiquement)",
        "",
        f"Jeu : {out['dataset']['n_rows']} lignes, {out['dataset']['n_fraud']} fraudes. "
        f"Modèle retenu (coût de validation) : **{r[out['champion']]['name']}**.",
        "",
        "| Modèle | Seuil | Précision | Rappel | F1 | FPR | FNR | ROC-AUC | PR-AUC | "
        "Accuracy | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ("dummy", "logreg", "tree"):
        t = r[m]["test"]
        lines.append(
            f"| {r[m]['name']} | {r[m].get('threshold', '-')} | {_fmt(t['precision'])} | "
            f"{_fmt(t['recall'])} | {_fmt(t['f1'])} | {_fmt(t['fpr'], True)} | "
            f"{_fmt(t['fnr'], True)} | {_fmt(t['roc_auc'])} | {_fmt(t['pr_auc'])} | "
            f"{_fmt(t['accuracy'], True)} | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} |"
        )
    (config.REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.FULL_DATASET))
    main(ap.parse_args().data)
