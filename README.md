# AI-TestGuard — qualification d'un modèle de détection de fraude bancaire

[![tests](https://github.com/deschantkn/AI-TestGuard/actions/workflows/tests.yml/badge.svg)](https://github.com/deschantkn/AI-TestGuard/actions/workflows/tests.yml)

Stratégie de test complète (QA pour l'IA) d'un classifieur de transactions
frauduleuses entraîné sur le jeu *Credit Card Fraud Detection* (Kaggle / ULB).
L'objectif n'est pas seulement d'obtenir un bon score. Il s'agit de **montrer, par des
tests reproductibles, dans quelles conditions le modèle est acceptable**.

## Structure

```
app/            code du pipeline
  preprocessing.py  contrat de données (validate_data), préparation (preprocess_data), split
  training.py       pipeline sklearn, train_model, réglage et choix du seuil (validation)
  prediction.py     predict_fraud / predict_proba, sauvegarde et chargement vérifiés (SHA-256)
  evaluation.py     evaluate_model : précision, rappel, F1, ROC-AUC, PR-AUC, FPR, FNR
  robustness.py     perturbations réalistes et mesure de sensibilité
  regression.py     comparaison à la baseline figée
  monitoring.py     PSI (dérive des données), taux d'alertes
  api.py            point de scoring : validation stricte, journaux sans données sensibles
  config.py         contrat, graines, coûts, critères d'acceptation (fixés a priori)
tests/          142 tests pytest (unitaires, intégration, robustesse, régression, sécurité, perf)
scripts/        expériences et rapports (évaluation, robustesse, perf, biais, monitoring)
baseline/       baseline.json : référence de non-régression (créée manuellement)
data/           échantillon versionné + README (le jeu complet n'est pas versionné)
reports/        métriques JSON, figures, rapport pytest
docs/           plan de test et rapports rédigés (dont le rapport final de qualification)
.github/workflows/tests.yml   CI : Ruff + Pytest + couverture
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests rapides (comme la CI, sans le jeu complet)
pytest -m "not slow" --cov=app

# Jeu complet (voir data/README.md), puis campagne complète
curl -sSL -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
export PYTHONPATH=.
python scripts/run_experiment.py        # comparaison des modèles, figures, modèle sauvegardé
python scripts/robustness_report.py     # scénarios de robustesse
python scripts/compare_versions.py      # non-régression contre baseline/baseline.json
python scripts/performance_report.py    # latence, P95, débit, mémoire
python scripts/bias_audit.py            # disparités par segment opérationnel
python scripts/monitoring_simulation.py # alertes de dérive simulées
python scripts/temporal_validation.py   # validation hors période
pytest -m slow                          # non-régression sur le jeu complet
```

## Documents

| Livrable | Fichier |
|---|---|
| Plan de test et matrice de traçabilité | [docs/01_plan_de_test.md](docs/01_plan_de_test.md) |
| Rapport d'évaluation ML | [docs/02_rapport_evaluation_ml.md](docs/02_rapport_evaluation_ml.md) |
| Robustesse et non-régression | [docs/03_rapport_robustesse_non_regression.md](docs/03_rapport_robustesse_non_regression.md) |
| Performance et sécurité opérationnelle | [docs/04_rapport_performance_securite.md](docs/04_rapport_performance_securite.md) |
| Audit des biais | [docs/05_audit_biais.md](docs/05_audit_biais.md) |
| Intégration continue et preuves | [docs/06_integration_continue.md](docs/06_integration_continue.md) |
| Surveillance après déploiement | [docs/07_surveillance.md](docs/07_surveillance.md) |
| **Rapport final de qualification** | [docs/08_rapport_final_qualification.md](docs/08_rapport_final_qualification.md) |
