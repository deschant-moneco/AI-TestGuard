# Rapport final de qualification : modèle de détection de fraude bancaire

**Projet** : AI-TestGuard · **Dépôt** : <https://github.com/deschantkn/AI-TestGuard> · **Version qualifiée** : baseline v1.0 (code `f2338df`, régression logistique, seuil 0,97) · **Date** : 22 novembre 2026

---

## Synthèse

| Question | Réponse |
|---|---|
| Le pipeline est-il correctement testé ? | **Oui.** 142 tests automatisés (unitaires, intégration, robustesse, non-régression, sécurité, performance, surveillance), 96 % de couverture, CI opérationnelle qui bloque les régressions (preuve : PR #1) |
| Les résultats sont-ils reproductibles ? | **Oui.** Métriques identiques au bit près entre macOS (local) et Linux (CI) ; deux entraînements donnent un écart de 0 ; prédictions identiques après sauvegarde et rechargement |
| Y a-t-il une fuite de données ? | **Non détectée.** Doublons supprimés, jeux disjoints, transformations ajustées sur le train seul, test utilisé une seule fois (tests T05, T09) |
| Le modèle atteint-il les critères d'acceptation ? | **Non.** Rappel 0,80 ✅, FPR 0,14 % ✅, ROC-AUC 0,96 ✅, mais **précision 0,494 ❌** (< 0,50) et **PR-AUC 0,678 ❌** (< 0,70) |
| Performance et sécurité opérationnelles ? | **Conformes** pour un prototype : P95 de 1,6 ms (modèle) et 3,2 ms (API locale), 2,3 M prédictions/s, modèle de 4 Ko, contrat d'entrée strict, journaux sans donnée sensible |
| **Décision** | **Rejeté pour une décision automatique. Accepté sous conditions pour un pilote en mode observation (shadow), sans blocage de transaction.** Détail en § 10 |

---

## 1. Contexte et périmètre de l'évaluation

Une banque veut qualifier un modèle de détection des transactions frauduleuses avant sa mise en production. Deux types d'erreurs comptent : les **faux positifs** (client légitime bloqué : réclamations, friction) et les **faux négatifs** (fraude non détectée : pertes financières).

* **Données** : jeu *Credit Card Fraud Detection* (ULB / Worldline), 284 807 transactions de cartes européennes sur 2 jours de septembre 2013, dont 492 fraudes (0,173 %). Variables : `Time`, `Amount` et 28 composantes PCA anonymisées (`V1`–`V28`).
* **Dans le périmètre** : contrat de données, prétraitement, entraînement de deux modèles légers (régression logistique, arbre de décision), choix du seuil, sauvegarde et rechargement, point de scoring (prototype), CI, surveillance (simulée).
* **Hors périmètre** : déploiement réel, authentification et TLS, tests de charge distribués, attaques adverses, équité envers des personnes (données absentes, voir § 9).

## 2. Plan de test et matrice de traçabilité

Le plan complet est dans [01_plan_de_test.md](01_plan_de_test.md). Il compte 21 exigences : les 7 imposées et 14 ajoutées, chacune reliée à un risque, à des cas de test nommés et à un résultat attendu.

| Exigence | Risque maîtrisé | Preuve | Statut |
|---|---|---|---|
| T01 colonnes présentes · T02 types · T08 cible ∈ {0, 1} | Scoring sur des données corrompues | `test_preprocessing.py` (38 tests) | ✅ |
| T03 aucune valeur non finie | Prédiction invalide | Tests unitaires + Hypothesis ; a détecté la régression de la PR #1 | ✅ |
| T04 probabilité valide | Décision incohérente | Test de propriété sur 80 entrées aléatoires | ✅ |
| T05 test indépendant · T09 découpage sans fuite | Score optimiste | `test_test_set_never_used_for_threshold_or_tuning` (on permute les labels du test : le seuil ne change pas) | ✅ |
| T06 reproductibilité · T12 sauvegarde et rechargement | Résultats non rejouables | Écart 0 ≤ 1e-9 | ✅ |
| T07 CI | Régression fusionnée | Runs GitHub, PR #1 bloquée | ✅ |
| T10 seuil par coût · T11 métriques exactes | Seuil arbitraire, accuracy trompeuse | `test_prediction.py` | ✅ |
| T13 intégrité du modèle · T17 contrat API · T18 journaux et secrets | Code arbitraire, injection, fuite | `test_integration.py`, `test_security.py` | ✅ |
| T14 entrées imparfaites · T15 sensibilité bornée | Plantage, instabilité | `test_robustness.py` + rapport sur 17 scénarios | ✅ avec anomalies A-03 et A-04 |
| T16 non-régression | Dégradation silencieuse | Baseline figée, 3 versions candidates comparées | ✅ |
| T19 performance | Latence incompatible | `test_performance.py` + protocole | ✅ |
| T20 détection de dérive | Dégradation en production | PSI et taux d'alertes | ✅ avec anomalie A-05 |
| **T21 critères de performance sur le test** | Modèle insuffisant en production | `reports/metrics.json` | ❌ |

## 3. Architecture du pipeline et choix méthodologiques

```
load_dataset ─► validate_data ─► split_data (doublons supprimés, stratifié 60/20/20, graine 42)
                                      │ train                     │ validation                  │ test
                                      ▼                           ▼                             ▼
Pipeline[ preprocess_data ─► SimpleImputer(median) ─► StandardScaler ─► LogisticRegression ]
          (sans état)        (ajusté sur train)       (ajusté sur train)  (C réglé sur val.)
                                                                         seuil = argmin coût(val.)
                                      └─► save_model (joblib + .meta.json SHA-256) ─► load_model (vérifié) ─► évaluation finale (test, une fois)
```

| Choix | Justification |
|---|---|
| `log1p(Amount)` ; heure cyclique (sin et cos) au lieu de `Time` brut | Montant très asymétrique ; `Time` brut est monotone et n'a pas de sens hors du jeu |
| V1–V28 conservées sans interprétation | Composantes PCA anonymisées : on ne leur attribue aucun sens métier |
| `class_weight="balanced"` | Rapport de classes 1:580 ; évite un modèle qui prédit toujours « normale » |
| Imputation et standardisation **dans** le `Pipeline` | Impossible de les ajuster sur le test par erreur ; un seul objet à sauvegarder |
| Choix des hyperparamètres sur la PR-AUC de validation | Métrique pertinente quand la classe positive est rare |
| Seuil par coût (FN 150 €, FP 5 €) sur la validation | Traduit l'asymétrie métier ; le seuil 0,5 n'a pas de sens avec des probabilités pondérées |
| Rejet ou correction explicite | Contrat : on **rejette** un schéma ou un type invalide ; on **corrige** NaN et inf (médiane), uniquement en batch |
| Baseline créée manuellement, jamais recalculée par la CI | Une régression ne doit pas devenir la nouvelle référence |

## 4. Résultats des tests unitaires et d'intégration

* **142 tests, 142 réussis** (localement, y compris le test lent sur le jeu complet) ; 141 en CI rapide sur Python 3.12 et 3.13. Rapport : [`reports/pytest_report.txt`](../reports/pytest_report.txt).
* **Couverture : 96 %** des lignes de `app/` (seuil CI : 85 %).
* **Ruff** : 0 erreur.
* Tests d'intégration clés :
  * scénario de bout en bout en 9 étapes, du chargement au rapport ;
  * prédictions identiques après rechargement ;
  * imputeur et scaler ajustés sur le train uniquement (moyennes vérifiées) ;
  * fichier de modèle altéré refusé ;
  * entrées invalides rejetées par le pipeline rechargé.

## 5. Métriques de performance et matrices de confusion

Détails, courbes et analyse : [02_rapport_evaluation_ml.md](02_rapport_evaluation_ml.md).

| Modèle (test) | Précision | Rappel | F1 | FPR | FNR | ROC-AUC | PR-AUC | Accuracy | Coût |
|---|---|---|---|---|---|---|---|---|---|
| Référence naïve | 0 | 0 | 0 | 0 % | 100 % | 0,50 | 0,002 | 99,83 % | 14 250 € |
| **Régression logistique (retenue)** | 0,494 | **0,800** | 0,610 | 0,138 % | 20,0 % | **0,962** | 0,678 | 99,83 % | **3 240 €** |
| Arbre de décision | 0,782 | 0,716 | 0,747 | 0,034 % | 28,4 % | 0,894 | 0,690 | 99,92 % | 4 145 € |

Matrice de confusion du modèle retenu (test) : **TP = 76, FP = 78, FN = 19, TN = 56 573**.

* La référence naïve a la **même accuracy** que le modèle retenu, ce qui confirme que l'accuracy ne mesure pas ici la capacité à détecter la fraude.
* IC95 sur le test : rappel [0,72 ; 0,87], précision [0,41 ; 0,57].
* **Validation hors période** (découpage chronologique) : rappel 0,81, précision 0,46, PR-AUC 0,74. Le classement des modèles se maintient, mais la précision reste sous 0,50.
* **Surapprentissage** : aucun pour la RL (PR-AUC train 0,73, validation 0,79) ; net pour l'arbre (ROC-AUC 0,999 en train contre 0,894 en test).

## 6. Robustesse et non-régression

Détails : [03_rapport_robustesse_non_regression.md](03_rapport_robustesse_non_regression.md).

* **17 scénarios** couvrant les 6 familles imposées. Tous ont le comportement prévu (rejet, traitement ou alerte), avec trois réserves :
  * rejet du **lot entier** dès qu'une seule ligne est invalide (A-03) ;
  * absence d'alerte sur les valeurs imputées et les montants atypiques ; l'imputation peut faire basculer une fraude, avec un Δ de probabilité allant jusqu'à 0,99 (A-04) ;
  * dérive non vue par le seul taux d'alertes (A-05).
* **Sensibilité au bruit** : au plus 0,02 % de changements de classe et Δ rappel ≤ 1 fraude à ε = 10 % σ.
* **Montant = 1 M€** : la précision tombe de 0,49 à 0,33.
* **Non-régression** :
  * baseline rejouée : `accept` ;
  * v1.1 (C = 1) : `accept` ;
  * v1.2 (arbre) : `reject` (rappel −0,106) ;
  * v1.3 (sans pondération) : `investigate` (seuil en bordure de grille).

## 7. Latence et consommation de ressources

Détails et protocole : [04_rapport_performance_securite.md](04_rapport_performance_securite.md).

| Mesure | Résultat | Seuil | |
|---|---|---|---|
| P95 d'une prédiction avec validation | 1,55 ms | ≤ 10 ms | ✅ |
| P95 d'une requête HTTP locale | 3,21 ms | ≤ 50 ms | ✅ |
| Débit en batch | 2,3 M / s | ≥ 20 000 / s | ✅ |
| Taille du modèle | 4,2 Ko | ≤ 5 Mo | ✅ |
| Pic mémoire pour scorer 56 746 lignes | 56 Mo | ≤ 200 Mo | ✅ |

Mesures réalisées sur Apple M3 Pro, sur une machine non dédiée, sans charge concurrente ni réseau réel.

## 8. Anomalies : gravité et statut

| ID | Gravité | Description | Source | Statut |
|---|---|---|---|---|
| **A-01** | **Haute** | Le modèle retenu ne respecte pas 2 critères d'acceptation sur le test : précision 0,494 (< 0,50) et PR-AUC 0,678 (< 0,70) | Évaluation finale | **Ouverte** (bloquante pour une décision automatique) |
| A-02 | Moyenne | PR-AUC en baisse entre validation et test (0,79 → 0,68). Le seuil retenu se trouve juste avant une forte chute de précision sur la courbe PR, ce qui rend la précision sensible au seuil. Effectifs faibles (94 et 95 fraudes) | Évaluation | Ouverte |
| A-03 | Moyenne | Une ligne invalide fait rejeter le lot entier | Robustesse 2b, 3c | Ouverte |
| A-04 | Moyenne | Pas de signalement des transactions imputées ni des montants atypiques ; FPR 5 fois plus élevé pour les montants ≥ 500 € (0,69 %, significatif) ; borne de rejet de 1 M€ trop laxiste (maximum observé : 25 691 €) | Robustesse 1, 3 ; audit par segment | Ouverte |
| A-05 | Moyenne | Le taux d'alertes seul ne détecte pas certaines dérives | Robustesse 5a, surveillance J2 | Atténuée (surveillance combinée PSI et taux d'alertes) |
| A-06 | Faible | PSI sur `Time` brut toujours en alerte (variable monotone) | Surveillance J3 | Atténuée (surveiller l'heure de la journée) |
| A-07 | Moyenne | `numpy==2.5.5` incompatible avec Python 3.11 : installation impossible | CI, run 45078564558 | **Corrigée** (matrice 3.12 et 3.13, `requires-python >= 3.12`) |
| A-08 | Faible | Probabilités non calibrées : seuils optimaux à 0,97–0,99, et 0,01 pour la v1.3, en bordure de grille | Choix du seuil, non-régression | Ouverte |
| A-09 | Faible | La protection de `main` ne s'applique pas aux administrateurs | CI | Acceptée (projet individuel) |
| A-10 | Moyenne | Surapprentissage de l'arbre de décision | Évaluation | Close (modèle non retenu) |

## 9. Limites de l'évaluation et risques résiduels

**Limites du jeu de données**

* **Ancienneté et représentativité.** Les données datent de 2013 : 2 jours, une seule zone (Europe). Les fraudes actuelles (paiement mobile, ingénierie sociale, 3-D Secure) ne sont pas représentées. **Les résultats ne prouvent rien, à eux seuls, sur les transactions bancaires actuelles.**
* **Petits effectifs.** Le test ne compte que 95 fraudes, et les intervalles de confiance sont larges : la précision se situe entre 0,41 et 0,57. Les écarts au seuil (A-01) ne sont pas statistiquement tranchés, dans un sens comme dans l'autre.
* **Variables anonymisées.** L'ACP et les variables d'origine inconnues empêchent toute explicabilité métier et tout contrôle de cohérence des variables.
* **Équité.** Aucun attribut de groupe n'est disponible : l'équité envers des personnes est **non évaluée** ([05_audit_biais.md](05_audit_biais.md)). Seule une disparité par segment de montant est démontrée.

**Limites méthodologiques**

* Les coûts de 150 € par FN et 5 € par FP sont des hypothèses, pas des chiffres validés par le métier.
* Les perturbations sont synthétiques, et le comportement face à un fraudeur adverse n'a pas été testé.
* Pas de calibration des probabilités, et seulement deux familles de modèles légers ont été comparées.
* La performance a été mesurée en local, sans réseau, sans TLS et sans concurrence.

**Risques résiduels si le modèle était déployé en l'état**

| Risque | Ordre de grandeur |
|---|---|
| Fraudes non détectées | Environ 1 sur 5 |
| Fausses alertes | Environ 1 sur 2 (environ 1 client sur 725 bloqué à tort, 1 sur 145 pour les montants ≥ 500 €) |
| Fraude non vue pendant plusieurs semaines | Délai d'arrivée des labels (contestations) |
| Dérive des données et dérive du concept | Inévitables avec le temps |

## 10. Décision argumentée

### 10.1 Décision technique

En application des critères fixés **avant** l'évaluation (plan de test, § 1.5) :

* critère 1 (tous les tests automatisés passent) : ✅ ;
* critère 2 (critères de performance sur le test) : ❌, précision et PR-AUC ;
* critère 3 (borne basse de l'IC95 du rappel ≥ 0,65) : ✅ (0,72) ;
* critère 4 (aucune anomalie haute ouverte) : ❌ (A-01) ;
* critère 5 (non-régression) : ✅ `accept`.

→ **Le modèle est REJETÉ pour une mise en production avec décision automatique (blocage de carte).**

Le critère n'est **pas** abaissé après coup. Remonter le seuil améliorerait la précision sur le test, mais choisir le seuil en regardant le test invaliderait l'évaluation.

### 10.2 Conditions pour un pilote

Le système **peut** être testé dans un environnement pilote, à ces conditions :

1. **Mode observation (shadow) ou aide à la décision uniquement.** Le modèle score en parallèle des règles existantes. Ses alertes vont à des analystes, sans blocage automatique. Le coût d'un faux positif devient alors un simple examen manuel, compatible avec une précision d'environ 0,5 (environ 0,27 % des transactions examinées).
2. **Anomalies moyennes corrigées avant le pilote** :
   * A-03 : quarantaine ligne par ligne ;
   * A-04 : signalement `degraded` des transactions imputées, alerte sur les montants > P99,9, borne de rejet ramenée à une valeur réaliste.
3. **Surveillance active dès le premier jour** (doc 07) : PSI, taux d'alertes, qualité des entrées, précision des alertes chaque semaine, échantillonnage aléatoire des transactions non alertées pour estimer le rappel.
4. **Critères de sortie du pilote fixés à l'avance** : sur au moins 3 mois, au moins 500 fraudes étiquetées, précision ≥ 0,50 et rappel ≥ 0,75 avec une **borne basse de l'IC95** au-dessus du seuil ; FPR des montants ≥ 500 € ≤ 0,5 %.
5. **Pistes d'amélioration**, à qualifier par la même chaîne de tests et de non-régression avant de remplacer la baseline :
   * calibration des probabilités (A-08) ;
   * grille de seuils plus fine ;
   * modèle v1.3 (sans pondération), évalué `investigate` ;
   * modèles plus expressifs (gradient boosting).

### 10.3 Ce qui reste nécessaire avant une mise en production réelle

**Réussir les tests techniques ne suffit pas pour être autorisé à déployer.** Il faut en plus :

| Domaine | Preuves supplémentaires exigées |
|---|---|
| **Données** | Réentraînement et requalification sur des **données récentes et internes** à la banque (plusieurs mois, tous les canaux), avec un jeu de test hors période d'au moins 500 fraudes |
| **Métier** | Validation des coûts FN et FP par la direction des risques ; résultats du pilote (§ 10.2) ; capacité de l'équipe d'analystes |
| **Équité et conformité** | Audit par groupe (Fairlearn) avec des attributs légalement utilisables ; analyse d'impact (AIPD, RGPD art. 35) ; respect de l'art. 22 du RGPD (intervention humaine, possibilité de contester) ; classification et obligations au titre de l'AI Act ; validation par le DPO et la conformité |
| **Sécurité** | Passerelle d'API avec authentification, TLS et limitation de débit ; modèle signé et stocké en lecture seule ; test d'intrusion ; secret scanning ; tests adverses (évasion) |
| **Opérations** | Tests de charge (Locust ou JMeter) sur l'infrastructure cible, avec la concurrence attendue en pointe ; SLO de disponibilité ; procédure de retour arrière ; supervision Prometheus et Grafana opérationnelle ; procédure de réentraînement et de mise à jour de la baseline avec revue humaine |
| **Gouvernance** | Fiche de modèle (*model card*), registre des versions, comité de validation des modèles (exigences de gestion du risque de modèle) |

---

## Question de synthèse

> *Dans quelles conditions votre modèle peut-il être considéré comme suffisamment fiable pour être testé dans un environnement pilote, et quelles preuves supplémentaires seraient nécessaires avant une mise en production réelle ?*

**Ce que nos résultats établissent.** La régression logistique retenue est techniquement **saine** :

* pipeline testé (142 tests, 96 % de couverture) ;
* résultats reproductibles au bit près sur deux systèmes d'exploitation ;
* aucune fuite de données détectée ;
* entrées imparfaites gérées de façon contrôlée ;
* rapidité largement suffisante (P95 < 2 ms) ;
* régressions bloquées par la CI.

Elle détecte **80 % des fraudes** (IC95 [0,72 ; 0,87]) en ne signalant que **0,27 % des transactions**. En revanche, elle ne satisfait **pas** deux des cinq critères métier fixés à l'avance : une alerte sur deux est fausse (précision 0,494) et la PR-AUC est de 0,678. Elle est aussi moins fiable sur les gros montants (FPR 5 fois plus élevé). Les données de 2013, sur 2 jours et avec 95 fraudes de test, ne permettent pas de trancher ces écarts ni de conclure sur l'équité.

**Elle peut donc être testée en pilote** à condition :

1. de fonctionner en **mode observation ou d'aide à l'analyste**, sans blocage automatique. Le coût d'une fausse alerte reste alors limité, et la précision observée devient acceptable ;
2. d'avoir corrigé auparavant la quarantaine ligne par ligne et le signalement des entrées dégradées et des montants atypiques ;
3. d'être sous **surveillance continue** : PSI et taux d'alertes immédiatement, précision et rappel au fil de l'arrivée des labels ;
4. d'avoir des **critères de sortie du pilote fixés à l'avance** et évalués avec leurs intervalles de confiance.

**Avant une mise en production réelle**, il faudrait en plus :

* une requalification complète sur des **données internes récentes** (au moins 500 fraudes hors période), avec des probabilités calibrées et la même chaîne de non-régression ;
* les **résultats du pilote**, montrant précision ≥ 0,50 et rappel ≥ 0,75 avec des intervalles de confiance au-dessus des seuils ;
* un **audit d'équité** par groupe sur des attributs légalement utilisables ;
* des **tests de charge et de sécurité** sur l'infrastructure cible : authentification, TLS, modèle signé, tests adverses ;
* une **validation métier, opérationnelle, sécuritaire et réglementaire** : coûts validés par les risques, AIPD et art. 22 du RGPD, AI Act, comité de validation des modèles.

Le système est **techniquement qualifié pour un pilote sous conditions. Il n'est pas autorisé à être déployé.**
