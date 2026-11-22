# 6. Intégration continue : workflow et preuves d'exécution

*Activité 9.*

* **Dépôt** : <https://github.com/deschantkn/AI-TestGuard>
* **Workflow** : [`.github/workflows/tests.yml`](../.github/workflows/tests.yml)
* **Historique des exécutions** : <https://github.com/deschantkn/AI-TestGuard/actions>

## 6.1 Organisation du workflow

| Job | Déclencheur | Contenu | Durée |
|---|---|---|---|
| `quick` : Lint + tests rapides (matrice Python 3.12 et 3.13) | Chaque `push` sur `main`, chaque **pull request**, manuel | `pip install -r requirements.txt` → **Ruff** → **Pytest** `-m "not slow"` (141 tests) avec **couverture** (`--cov-fail-under=85`) et rapport JUnit → résumé de couverture dans la page du run → artefacts `coverage.xml` et `junit.xml` | ≈ 35–45 s |
| `full` : Campagne complète | **Planifié** (lundi 03:00 UTC) ou **manuel** (`workflow_dispatch`, option `full=true`) | Téléchargement du jeu complet et **vérification SHA-256**, entraînement et évaluation, robustesse, non-régression, performance, test lent `-m slow`, archivage de `reports/` | ≈ 5 min |

Choix de conception :

* Les **tests rapides** s'exécutent à chaque PR sur l'échantillon versionné de 5 000 lignes. Les entraînements complets, les mesures de performance et la non-régression sur le jeu complet sont **coûteux** : ils passent dans la campagne manuelle ou hebdomadaire.
* `fail-fast: false` : un échec sur une version de Python n'interrompt pas l'autre, ce qui donne un diagnostic complet.
* `concurrency` annule les runs devenus obsolètes sur la même branche. Effet de bord observé : un `workflow_dispatch` lancé juste avant un push sur `main` a été annulé (run 45078675208), puis relancé.
* `permissions: contents: read` : principe du moindre privilège, aucun secret nécessaire.
* Un échec de test fait échouer le job (code de sortie non nul de pytest) : la PR est marquée ❌ et **la fusion est bloquée** par la protection de branche.

## 6.2 Protection de la branche principale

Configurée par l'API GitHub (`PUT /repos/…/branches/main/protection`) :

* contrôles obligatoires : `Lint + tests rapides (Python 3.12)` et `Lint + tests rapides (Python 3.13)` ;
* `strict: true` : la branche doit être à jour avec `main` avant la fusion ;
* limite : `enforce_admins: false`. Le propriétaire du dépôt peut encore pousser directement sur `main`, ce qui a servi à corriger la matrice Python. C'est acceptable pour un projet individuel. En équipe, il faudrait activer `enforce_admins` et exiger une revue de PR (anomalie A-09).

## 6.3 Preuves d'exécution

| # | Run | Événement | Résultat | Ce que cela prouve |
|---|---|---|---|---|
| 1 | [45078564558](https://github.com/deschantkn/AI-TestGuard/actions/runs/45078564558) | push `main` (premier commit) | ❌ Python 3.11 · ✅ Python 3.12 | **Échec réel détecté par la CI** : `numpy==2.5.5` exige Python ≥ 3.12 (anomalie A-07). Corrigé en passant la matrice à 3.12 et 3.13 |
| 2 | [45078708564](https://github.com/deschantkn/AI-TestGuard/actions/runs/45078708564) | push `main` (correctif) | ✅ 3.12 · ✅ 3.13 | **Exécution réussie** : Ruff OK, 141 tests passés, couverture ≥ 85 % |
| 3 | [45078828351](https://github.com/deschantkn/AI-TestGuard/actions/runs/45078828351) / [PR #1](https://github.com/deschantkn/AI-TestGuard/pull/1) | pull request | ❌ 3.12 · ❌ 3.13 | **Échec volontaire correctement détecté** (détails ci-dessous) |
| 4 | [45078911927](https://github.com/deschantkn/AI-TestGuard/actions/runs/45078911927) | `workflow_dispatch` (full=true) | ✅ quick · ✅ full (5 min) | Campagne complète sur le jeu Kaggle : empreinte du jeu vérifiée, métriques **identiques au bit près** à l'exécution locale macOS (rappel 0,8, précision 0,49350…, ROC-AUC 0,96212…), non-régression `accept`, test lent passé |

### Détail de l'échec volontaire (PR #1)

La PR supprime la conversion `±inf → NaN` dans `preprocess_data` :

```diff
-    x = df[config.FEATURE_COLUMNS].astype("float64").replace([np.inf, -np.inf], np.nan)
+    x = df[config.FEATURE_COLUMNS].astype("float64")  # RÉGRESSION VOLONTAIRE : inf non traité
```

Résultat de la CI, identique sur les deux versions de Python :

```
FAILED tests/test_preprocessing.py::test_infinite_values_are_reported_and_converted_to_nan - AssertionError: assert not np.True_
FAILED tests/test_preprocessing.py::test_pipeline_imputation_yields_finite_output - ValueError: Input X contains infinity or a value too large for dtype('float64').
FAILED tests/test_robustness.py::test_infinite_values_are_treated - ValueError: Input X contains infinity or a value too large for dtype('float64').
================= 3 failed, 138 passed, 1 deselected =================
```

* L'exigence **T03** (aucune valeur non finie après le prétraitement) et le scénario de robustesse 2 ont détecté la régression.
* État de la PR : `mergeStateStatus = BLOCKED`, la fusion est impossible. La PR a été fermée sans fusion et reste visible comme preuve.

## 6.4 Exécution locale équivalente

```bash
ruff check .
pytest -m "not slow" --cov=app --cov-report=term-missing   # 141 passed, couverture 96 %
pytest -m slow                                              # non-régression sur le jeu complet
```

Rapport d'exécution complet (142 tests, y compris le test lent) : [`reports/pytest_report.txt`](../reports/pytest_report.txt) et `reports/junit.xml`.

| Module | Couverture |
|---|---|
| `app/config.py`, `evaluation.py`, `regression.py` | 100 % |
| `app/preprocessing.py` | 98 % |
| `app/training.py` | 98 % |
| `app/monitoring.py` | 97 % |
| `app/prediction.py` | 96 % |
| `app/api.py`, `app/robustness.py` | 95 % |
| `app/experiment.py` | 89 % (repli sans git non exercé) |
| **Total** | **96 %** |

> **Note** : le dépôt a d'abord été créé sous le compte `deschant-moneco`, puis transféré vers `deschantkn`. Les identifiants de runs, la PR #1 et la protection de branche sont conservés lors d'un transfert GitHub. Les anciennes URL redirigent vers les nouvelles.
