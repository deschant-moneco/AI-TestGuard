# 1. Plan de test : système de détection de fraude bancaire

*Activité 1 : ce qu'on teste, pourquoi, comment et avec quels critères.*

## 1.1 Périmètre du système

```
CSV / requête JSON ──► validate_data ──► preprocess_data ──► SimpleImputer ──► StandardScaler ──► Modèle ──► probabilité ──► seuil ──► décision
   (chargement)        (contrat :        (sans état :       (médiane         (moyenne/σ        (RL ou      [0,1]        (choisi     0 / 1
                        rejet)            log1p, heure)      apprise TRAIN)   appris TRAIN)      arbre)                  sur VAL)
                                          └────────────── sklearn.Pipeline sauvegardé (joblib + SHA-256) ─────────────┘
```

| Composant | Contenu | Hors périmètre |
|---|---|---|
| **Données** | Jeu Kaggle *Credit Card Fraud* (284 807 lignes, 492 fraudes), échantillon versionné de 5 000 lignes, contrat de schéma | Collecte en temps réel, qualité des sources amont |
| **Prétraitement** | Contrôle du schéma et des types, `log1p(Amount)`, heure cyclique, imputation, standardisation | Ingénierie de variables métier (V1–V28 anonymisées) |
| **Modèle** | Régression logistique et arbre de décision (`class_weight="balanced"`), réglage et seuil sur la validation | Modèles lourds (boosting, réseaux) |
| **Prédiction** | Probabilité, décision au seuil métier, sauvegarde et rechargement vérifiés | Explicabilité individuelle |
| **Intégration** | Pipeline de bout en bout, point de scoring JSON/HTTP (`app/api.py`), CI GitHub Actions | Déploiement réel, authentification, passerelle de paiement |

## 1.2 Hypothèses métier fixées **avant** l'évaluation

| Paramètre | Valeur | Justification |
|---|---|---|
| Coût d'un faux négatif | 150 € | Montant moyen d'une fraude dans le jeu (≈ 122 €) + frais de litige |
| Coût d'un faux positif | 5 € | Contrôle manuel ou SMS de confirmation, friction client |
| Règle de seuil | Minimiser le coût attendu **sur la validation** | Le test ne sert jamais au réglage |
| Rappel minimal | ≥ 0,75 | Au moins 3 fraudes sur 4 interceptées |
| Précision minimale | ≥ 0,50 | Au plus 1 alerte sur 2 injustifiée (capacité de l'équipe d'analystes) |
| FPR maximal | ≤ 0,5 % | Au plus 1 client légitime sur 200 bloqué |
| PR-AUC minimale | ≥ 0,70 | Qualité de classement sur la classe rare (référence aléatoire = 0,0017) |
| ROC-AUC minimale | ≥ 0,90 | Pouvoir discriminant global |
| Tolérances de non-régression | rappel 0,03 · précision 0,05 · PR-AUC 0,02 · ROC-AUC 0,01 · FPR 0,001 | Environ 2–3 fraudes de validation (94 fraudes au total) |
| Reproductibilité | écart absolu ≤ 1e-9 sur les probabilités | Graines fixées, calcul déterministe |

Ces valeurs sont codées dans `app/config.py` et versionnées dès le premier commit.

## 1.3 Exigences, risques et cas de test (matrice de traçabilité)

Type : D = données, U = unitaire, F = fonctionnel ML, ML = évaluation, NR = non-régression, CI, I = intégration, R = robustesse, S = sécurité, P = performance, M = surveillance.

| ID | Exigence | Type | Risque maîtrisé | Cas de test (fichier::test) | Résultat attendu | Statut |
|---|---|---|---|---|---|---|
| **T01** | Les colonnes nécessaires sont présentes | D | Scoring sur des données incomplètes, décalage silencieux des colonnes | `test_preprocessing::test_missing_required_column_is_rejected` (6 cas), `test_duplicated_column_is_rejected` | Aucune colonne indispensable manquante, sinon `DataValidationError` | ✅ |
| **T02** | Les données respectent les types attendus | U | Coercition implicite (`"12,5"`, booléens), cible corrompue | `test_non_numeric_feature_is_rejected`, `test_boolean_column_is_rejected`, `test_target_outside_0_1_is_rejected`, `test_missing_target_is_rejected` | Types conformes au contrat, sinon rejet | ✅ |
| **T03** | Le prétraitement ne produit pas de valeurs non finies | U | NaN ou inf propagés jusqu'au modèle : prédiction invalide ou plantage | `test_infinite_values_are_reported_and_converted_to_nan`, `test_pipeline_imputation_yields_finite_output`, `test_preprocess_output_is_finite_for_any_valid_amount_and_time` (Hypothesis) | Aucune valeur NaN ou infinie en sortie | ✅ |
| **T04** | Le modèle produit une probabilité valide | F | Score hors [0, 1] ou NaN, décision incohérente | `test_prediction::test_prediction_probabilities_are_valid`, `test_robustness::test_property_probability_always_valid` (80 entrées aléatoires) | Probabilité finie dans [0, 1] | ✅ |
| **T05** | Les performances sont évaluées sur des données indépendantes | ML | Score optimiste par fuite train → test | `test_integration::test_test_set_never_used_for_threshold_or_tuning`, `test_splits_are_disjoint_without_duplicate_rows`, `test_scaler_and_imputer_fitted_on_train_only` | Jeu de test jamais utilisé pour le réglage | ✅ |
| **T06** | Les résultats sont reproductibles | NR | Résultats non rejouables, régressions masquées | `test_regression::test_two_runs_are_identical`, `test_training_is_deterministic`, `test_split_is_reproducible_with_seed` | Écart ≤ 1e-9 | ✅ |
| **T07** | Les tests s'exécutent sur GitHub | CI | Régression fusionnée sans être vue | Workflow `tests.yml` ; PR #1 d'échec volontaire | Workflow réussi, ou échec signalé et fusion bloquée | ✅ |
| *Exigences ajoutées* | | | | | | |
| **T08** | Les valeurs de cible hors {0, 1} sont rejetées | D | Étiquettes corrompues : entraînement faussé | `test_target_outside_0_1_is_rejected` (2, −1, 0,5) | `DataValidationError` | ✅ |
| **T09** | Les doublons sont supprimés avant le découpage, les jeux sont disjoints et stratifiés | I | Fuite par doublons (1 081 dans le jeu) ; taux de fraude déséquilibré entre les jeux | `test_split_is_stratified_disjoint_and_deduplicated` ; `leakage_checks` dans `reports/metrics.json` | 0 recouvrement ; taux de fraude à ±0,03 | ✅ |
| **T10** | Le seuil est choisi sur la validation selon le coût métier | ML | Seuil arbitraire (0,5) ou réglé sur le test | `test_select_threshold_minimizes_cost`, `test_select_threshold_favours_recall_when_fn_expensive` | Seuil de coût minimal ; il baisse quand le coût d'un FN augmente | ✅ |
| **T11** | Les métriques sont exactes et adaptées au déséquilibre | U | Accuracy trompeuse ; erreur de calcul FPR/FNR | `test_evaluate_model_known_confusion_matrix`, `test_accuracy_is_misleading_on_imbalanced_data`, `test_auc_undefined_when_single_class` | Valeurs exactes ; un modèle à rappel nul est refusé | ✅ |
| **T12** | Le pipeline est identique après sauvegarde et rechargement | I | Divergence entre l'entraînement et la production | `test_predictions_identical_after_save_and_reload` | Écart ≤ 1e-9 ; décisions identiques | ✅ |
| **T13** | Un fichier de modèle altéré ou non conforme est refusé | S | Exécution de code arbitraire via pickle ; modèle substitué | `test_load_model_rejects_tampered_file`, `test_load_model_rejects_bad_extension_missing_file_and_metadata` | `ModelIntegrityError` | ✅ |
| **T14** | Les entrées imparfaites ont un comportement défini (rejet, traitement ou alerte) | R | Plantage ou décision silencieusement fausse | `test_robustness.py` (6 scénarios), `scripts/robustness_report.py` | Comportement conforme au tableau du rapport de robustesse | ✅ (avec anomalies A-03, A-04) |
| **T15** | Une petite perturbation a un effet borné et mesuré | R | Modèle instable près de la frontière de décision | `test_small_pca_noise_has_bounded_effect`, `test_sensitivity_increases_with_noise_amplitude` | ≤ 1 / 3 / 5 % de changements de classe pour ε = 1 / 5 / 10 % σ | ✅ |
| **T16** | Une version candidate n'est pas moins bonne que la baseline au-delà de la tolérance | NR | Dégradation introduite par un changement de code ou de paramètres | `test_no_regression_on_sample_validation`, `test_no_regression_on_full_validation` (lent), `scripts/compare_versions.py` | Décision `accept` / `investigate` / `reject` | ✅ |
| **T17** | L'API rejette les champs manquants ou en trop, les types incorrects et les charges excessives | S | Injection de champs, contournement du contrat, déni de service | `test_security.py` (12 fonctions, 22 cas) | HTTP 400 ou 413, message explicite | ✅ |
| **T18** | Aucune donnée sensible dans les journaux, aucun secret dans le dépôt | S | Fuite de données (RGPD, PCI-DSS) | `test_logs_do_not_leak_sensitive_values`, `test_no_secrets_in_repository`, `test_sensitive_files_are_gitignored` | Journaux avec un identifiant pseudonyme seulement ; aucun motif de secret trouvé | ✅ |
| **T19** | Latence P95, débit et taille du modèle respectent les seuils | P | Scoring trop lent pour l'autorisation en temps réel | `test_performance.py`, `scripts/performance_report.py` | P95 ≤ 10 ms (modèle), ≤ 50 ms (API) ; débit ≥ 20 000/s ; taille ≤ 5 Mo | ✅ |
| **T20** | Une dérive des entrées est détectée | M | Dégradation silencieuse en production | `test_covariate_shift_triggers_drift_alert`, `test_monitoring.py` | Alerte PSI ≥ 0,25 sur les variables décalées ; aucune alerte sans dérive | ✅ (anomalie A-05) |
| **T21** | Le modèle atteint les critères d'acceptation sur le test | ML | Mise en production d'un modèle insuffisant | `reports/metrics.json` → `acceptance_test` | Les 5 critères respectés | ❌ (voir rapport final) |

## 1.4 Outils et preuves conservées

| Outil | Usage | Preuve conservée |
|---|---|---|
| `pytest`, `pytest-cov` | 142 tests, couverture | `reports/pytest_report.txt`, `reports/junit.xml`, artefacts de la CI |
| `hypothesis` | Tests par propriétés (probabilité valide, prétraitement fini) | Exemples minimaux en cas d'échec |
| `scikit-learn` `Pipeline`, `joblib` | Pipeline unique, sauvegardé avec son empreinte SHA-256 | `models/*.meta.json` (versions, seuil, empreinte du jeu) |
| `ruff` | Style et erreurs statiques | Étape CI « Ruff » |
| `time.perf_counter`, `psutil`, `tracemalloc` | Latence, débit, mémoire | `reports/performance.json` |
| `matplotlib` | ROC, PR, matrices de confusion, coût selon le seuil | `reports/figures/*.png` |
| GitHub Actions | Exécution à chaque push et PR ; campagne complète manuelle ou hebdomadaire | Historique des runs, PR #1 |
| `baseline/baseline.json` | Référence figée de non-régression | Versionnée, ne change que par une action manuelle |

## 1.5 Critères d'acceptation et de rejet

**Le modèle est techniquement accepté pour un pilote si, et seulement si :**

1. tous les tests automatisés passent (T01–T20) ;
2. les 5 critères de performance (§ 1.2) sont respectés sur le jeu de **test**, avec un seuil fixé sur la validation ;
3. la borne basse de l'intervalle de confiance à 95 % du rappel de test est ≥ 0,65 (petit effectif : 95 fraudes) ;
4. aucune anomalie de gravité **haute** n'est ouverte ;
5. la non-régression contre la baseline donne `accept`.

**Rejet** si l'un des critères 1, 2 ou 5 échoue. Si seuls les critères 3 ou 4 échouent, la décision est « **investigation** » : pilote possible en mode observation (shadow), sans décision automatique.

Une acceptation technique **n'autorise pas** un déploiement : il faut en plus une validation métier, opérationnelle, de sécurité et réglementaire (voir le rapport final).

## 1.6 Répartition des tests

| Fichier | Cas exécutés | Contenu |
|---|---|---|
| `test_preprocessing.py` | 38 | Contrat de données, NaN et inf, valeurs limites, prétraitement, découpage, chargement |
| `test_prediction.py` | 31 | Entraînement, prédiction, seuil, métriques |
| `test_integration.py` | 11 | Bout en bout, sauvegarde et rechargement, fuites, intégrité |
| `test_robustness.py` | 21 | 6 familles de scénarios, propriétés (Hypothesis) |
| `test_regression.py` | 9 | Baseline, reproductibilité, logique de comparaison (dont 1 test lent) |
| `test_security.py` | 22 | Contrat API, journaux, HTTP, secrets |
| `test_performance.py` | 3 | Latence P95, débit, taille |
| `test_monitoring.py` | 7 | PSI, taux d'alertes |
| **Total** | **142** | |
