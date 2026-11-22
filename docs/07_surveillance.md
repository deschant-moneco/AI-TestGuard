# 7. Stratégie de surveillance après déploiement

*Activité 10. Code : `app/monitoring.py`. Simulation : `python scripts/monitoring_simulation.py` (résultats dans `reports/monitoring_simulation.json`).*

## 7.1 Dérive des données ou dérive du concept

| | **Dérive des données** (*data drift*) | **Dérive du concept** (*concept drift*) |
|---|---|---|
| Définition | La distribution des entrées P(X) change : nouveaux commerçants, saisonnalité, inflation des montants, changement de terminal | La relation P(fraude \| X) change : nouveaux modes opératoires, fraudeurs qui s'adaptent au modèle |
| Labels nécessaires ? | **Non**, observable immédiatement | **Oui**, il faut savoir quelles transactions étaient frauduleuses |
| Signaux | PSI par variable, statistiques de complétude, taux d'alertes, distribution des scores | Rappel, précision, FPR et FNR calculés sur les labels arrivés après coup |
| Délai de détection | Minutes à heures | Semaines : une contestation (*chargeback*) arrive en général 30 à 90 jours après la transaction |

Une dérive des données n'implique pas forcément une baisse de performance, et une dérive du concept peut survenir **sans** dérive des données visible. Il faut donc suivre les deux.

## 7.2 Indicateurs surveillés

| Indicateur | Disponibilité | Fréquence | Seuil d'attention | Seuil d'alerte | Action |
|---|---|---|---|---|---|
| **PSI par variable** (V1–V28, `log_amount`, **heure**) contre la référence du train | Immédiate | Quotidienne (fenêtre de 24 h) | PSI ≥ 0,10 | PSI ≥ 0,25 sur ≥ 3 variables, ou sur V4 / V14 / V12 / V10 (plus forts coefficients standardisés du modèle : +1,37 / −1,13 / −1,02 / −0,93) | Analyse, comparaison avec les labels dès qu'ils arrivent, réentraînement envisagé |
| **Taux de transactions signalées** | Immédiate | Horaire et quotidienne | Écart > × 1,5 | Hors de [÷2 ; ×2] du taux de validation (0,30 %) | Vérifier les entrées et le seuil ; risque de surcharge des analystes ou de fraude non vue |
| **Distribution des scores** (part > 0,5, médiane) | Immédiate | Quotidienne | PSI du score ≥ 0,10 | ≥ 0,25 | Comme ci-dessus |
| **Qualité des entrées** : taux de rejet, de NaN imputés, de montants > P99,9 | Immédiate | Temps réel | > 0,1 % | > 1 % | Incident sur les données en amont |
| **Disponibilité de l'API** | Immédiate | Continue | < 99,9 % sur 1 h | < 99,5 % | Astreinte |
| **Latence de l'API (P95, P99)** | Immédiate | Continue | P95 > 20 ms | P95 > 50 ms | Montée en charge, profilage |
| **Précision des alertes** (retour des analystes) | J+1 à J+7 | Hebdomadaire | < 0,45 | < 0,35 | Revoir le seuil sur la validation la plus récente |
| **Rappel, FNR** (contestations reçues) | J+30 à J+90 | Mensuelle | Rappel < 0,75 | Rappel < 0,65 | Réentraînement et nouvelle qualification (non-régression contre la baseline) |
| **FPR** (clients bloqués à tort, réclamations) | J+7 à J+30 | Mensuelle | > 0,3 % | > 0,5 % | Idem, avec examen par segment (montants ≥ 500 €) |

**Quand les labels sont retardés ou absents :**

* **Suivis tout de suite** : PSI, taux d'alertes, distribution des scores, qualité des entrées, disponibilité et latence.
* **Validés plus tard** : précision (retour des analystes, sous une semaine), FPR (réclamations), rappel et FNR (contestations, 1 à 3 mois).
* Le pic de fraude non détectée n'est donc visible qu'avec **un retard structurel**. D'où l'intérêt de suivre de près les signaux immédiats, et pendant le pilote, d'**échantillonner au hasard** des transactions non alertées pour les faire étiqueter.

## 7.3 Alertes simulées

Référence : distribution du train et taux d'alertes de validation (0,298 %).

| Flux simulé | Transactions | Variables en alerte PSI (≥ 0,25) | Variables en attention | PSI les plus élevés | Taux d'alertes | Alerte taux | Diagnostic |
|---|---|---|---|---|---|---|---|
| J1 : test (même distribution) | 56 746 | 0 | 0 | V25 0,0005 | 0,271 % | non | ✅ Aucune fausse alerte |
| J2 : V + 0,5σ, Amount × 3 | 56 746 | **29** | 0 | V28 1,99 ; V8 1,74 ; V27 1,70 | 0,196 % | **non** | ⚠️ Seul le PSI détecte la dérive (A-05) |
| J3 : seconde journée seule | 27 749 | 4 | 2 | **Time 6,10** ; V1 0,51 ; V3 0,33 | 0,332 % | non | ⚠️ `Time` brut est monotone et alerte toujours : c'est un artefact (A-06). Vraie dérive intrajournalière modérée sur V1 et V3 |
| J4 : V − 0,5σ | 56 746 | **27** | 1 | V1 3,17 ; V28 1,77 ; V20 1,62 | **0,740 %** | **oui** (× 2,5) | ✅ Double alerte : PSI et surcharge d'alertes |

**Enseignements.**

1. Le PSI ne génère pas de faux positif sur un flux de même distribution (J1) et détecte toutes les dérives injectées.
2. Le taux d'alertes seul **ne suffit pas** (J2) : il faut combiner les deux signaux.
3. `Time` (secondes depuis le début) ne doit **pas** être surveillé brut. On surveille l'**heure de la journée**, c'est-à-dire la variable réellement utilisée par le modèle (A-06).

## 7.4 Architecture proposée

* **Prototype (actuel)** : scripts planifiés (job `full` hebdomadaire de la CI), rapports JSON et alertes simulées.
* **Production** : chaque prédiction exporte ses métriques au format Prometheus (`fraud_score` en histogramme, `alerts_total`, `rejected_total`, `imputed_total`, latence) ; un job quotidien calcule le PSI ; tableaux de bord et règles d'alerte Grafana / Alertmanager ; table de labels alimentée par les contestations et le retour des analystes ; recalcul mensuel des métriques supervisées et de l'audit par segment.
* **Réentraînement** : jamais automatique. Une alerte déclenche une analyse. Un nouveau modèle repasse toute la qualification (tests et non-régression contre la baseline figée). La baseline n'est mise à jour qu'après une revue humaine.
