# 3. Rapport de robustesse et de non-régression

*Activités 5 et 6.2. Pour reproduire : `scripts/robustness_report.py`, `scripts/compare_versions.py`. Tests : `tests/test_robustness.py`, `tests/test_regression.py`. Chiffres bruts : `reports/robustness.json`, `reports/regression_comparison.json`.*

## 3.1 Robustesse : protocole

* Modèle évalué : la régression logistique retenue (seuil 0,97), rechargée depuis `models/fraud_model.joblib` avec contrôle SHA-256.
* Données : le **jeu de test complet** (56 746 lignes, 95 fraudes). Pour chaque scénario, on compare les prédictions sur les données perturbées à celles sur les données d'origine.
* Référence non perturbée : rappel 0,800, précision 0,494, FPR 0,138 %, taux d'alertes 0,27 %.
* Mesures : variation moyenne et maximale de la probabilité, nombre de changements de classe, Δ rappel, Δ précision, Δ FPR, nombre de lignes invalides.

**Justification des perturbations.** `V1`–`V28` sont des composantes PCA anonymisées d'échelles très différentes (σ de 0,33 à 2,0). On leur applique donc un bruit **proportionnel à l'écart-type de chaque composante** (mesuré sur le train), sans leur attribuer de sens métier. `Amount` (en euros) et `Time` (en secondes) reçoivent des perturbations propres à leur signification : variation relative du montant (frais, conversion), montants extrêmes, décalage horaire.

## 3.2 Tableau des scénarios et résultats

| # | Scénario | Comportement attendu | Comportement observé | Changements de classe | Δ proba moy. / max | Δ rappel | Δ précision | Δ FPR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1a | Manquants : V4, 5 % des lignes | Traiter (médiane) + avertissement | Traité, 2 873 NaN signalés | 2 (0,004 %) | 0,003 / 0,858 | 0 | +0,007 | 0 | ✅ |
| 1b | Manquants : V4 + V14 + Amount, 10 % des lignes | Traiter + avertissement | Traité | 5 (0,009 %) | 0,007 / **0,990** | 0 | +0,010 | −0,0001 | ⚠️ A-04 |
| 1c | Manquants : toutes les variables, 5 % des lignes | Traiter + alerte qualité | Traité | 4 | 0,003 / 0,977 | **−0,021** | 0 | 0 | ⚠️ A-04 |
| 2a | `+inf` sur V14, 5 % des lignes | Traiter (inf → NaN → médiane) | Traité, 2 857 inf signalés | 3 | 0,002 / 0,801 | 0 | −0,003 | 0 | ✅ |
| 2b | Texte « N/A » dans V10 (1 ligne) | Rejeter | **Tout le lot est rejeté** (56 746 lignes) | – | – | – | – | – | ⚠️ A-03 |
| 3a | Amount × 100 (plafonné à 1 M€) | Traiter | Traité | 24 | 0,020 / 0,098 | +0,011 | **−0,061** | +0,0004 | ✅ (sensibilité mesurée) |
| 3b | Amount = 1 000 000 € pour toutes les lignes | Traiter + alerte métier | Traité, **sans alerte** | 81 | 0,056 / 0,286 | +0,011 | **−0,166** | +0,0014 | ⚠️ A-04 |
| 3c | Amount × 10⁷ (> 1 M€) | Rejeter | Rejeté (« borne de rejet ») | – | – | – | – | – | ✅ (A-03) |
| 4a | Bruit V1–V28, ε = 1 % σ | Traiter, effet faible | 0 changement | 0 | 0,001 / 0,025 | 0 | 0 | 0 | ✅ |
| 4b | Bruit V1–V28, ε = 5 % σ | Traiter, effet borné | 10 changements | 10 (0,02 %) | 0,005 / 0,125 | +0,011 | 0 | 0 | ✅ |
| 4c | Bruit V1–V28, ε = 10 % σ | Traiter, effet borné | 14 changements | 14 (0,02 %) | 0,011 / 0,247 | +0,011 | −0,006 | +0,0001 | ✅ |
| 4d | Amount ± 5 % | Traiter | 0 changement | 0 | 0,0001 / 0,001 | 0 | 0 | 0 | ✅ |
| 4e | Time + 1 h | Traiter | 3 changements | 3 | 0,003 / 0,023 | +0,011 | −0,003 | 0 | ✅ |
| 5a | Dérive : V + 0,5σ et Amount × 3 | Alerte de dérive | 29 alertes PSI ; **taux d'alertes −34 % sans alarme** | 43 | 0,050 / 0,415 | −0,011 | +0,182 | −0,0007 | ⚠️ A-05 |
| 5b | Dérive : V − 0,5σ | Alerte de dérive | 27 alertes PSI + alarme du taux d'alertes (× 2,5) | 266 | 0,140 / 0,415 | +0,042 | **−0,303** | +0,0046 | ✅ |
| 6a | Colonne V1 absente | Rejeter | Rejeté | – | – | – | – | – | ✅ |
| 6b | Colonne `Amount` renommée en `amount` | Rejeter | Rejeté | – | – | – | – | – | ✅ |

Un changement de 1 sur 95 fraudes vaut 0,0105 de rappel. Les Δ rappel de ±0,011 correspondent donc à **une seule fraude**.

### Lecture des résultats

* **Sensibilité au bruit (scénario 4).** Elle augmente avec l'amplitude, ce que vérifie le test `test_sensitivity_increases_with_noise_amplitude`. Elle reste toutefois très faible : au plus 0,02 % de changements de classe et aucune dégradation du rappel à ε = 10 % σ. On n'impose pas une invariance stricte : les changements observés portent sur des transactions proches du seuil (Δ proba max de 0,25 à ε = 10 %).
* **Valeurs manquantes (1).** L'impact moyen est négligeable. En revanche, **l'imputation par la médiane peut faire basculer une transaction individuelle** (Δ proba jusqu'à 0,99) : une fraude dont V14, l'une des variables les plus influentes du modèle, est manquante devient « normale ». Le rappel baisse de 0,021 (2 fraudes) quand toutes les variables ont 5 % de manquants.
* **Montants extrêmes (3).** Le modèle reste valide numériquement, mais un montant élevé augmente le score. Avec Amount = 1 M€, la précision chute de 0,49 à 0,33. Aucune alerte métier n'est produite pour ces montants atypiques mais encore acceptés.
* **Dérive (5).** Le PSI détecte toujours la dérive. Le simple suivi du taux d'alertes la rate dans le scénario 5a, car le taux baisse de 34 % (par rapport à la validation), ce qui reste dans la bande × 2.
* **Schéma (6) et types (2b).** Tout est correctement rejeté avec un message explicite. Mais le rejet porte sur le **lot entier**.

## 3.3 Anomalies et limites (robustesse)

| ID | Gravité | Description | Recommandation | Statut |
|---|---|---|---|---|
| A-03 | Moyenne | Une seule ligne invalide fait rejeter tout le lot (56 746 transactions). En batch, une ligne corrompue bloque donc le scoring de toutes les autres. | Valider ligne par ligne et mettre les lignes invalides en quarantaine (sortie « invalides » séparée) ; rejet global seulement si le schéma est invalide | Ouverte |
| A-04 | Moyenne | Pas d'alerte quand une transaction scorée a des valeurs imputées ou un montant atypique. L'imputation peut faire basculer une fraude (Δ proba 0,99). Les montants très élevés dégradent la précision. | Marquer les transactions imputées (`degraded=True`) et les envoyer en revue manuelle ; ajouter une alerte au-delà du P99,9 des montants observés (≈ 3 000 €), et revoir la borne de rejet de 1 M€ (le maximum observé est 25 691 €) | Ouverte |
| A-05 | Moyenne | Le taux d'alertes seul ne détecte pas certaines dérives (5a) | Surveiller à la fois le PSI par variable et le taux d'alertes (voir doc 07) | Prise en compte dans la surveillance |

**Limites.** Les perturbations sont synthétiques : on ne connaît pas la vraie distribution des erreurs de saisie ou de capteur. Le bruit sur des composantes PCA ne correspond pas à une erreur physique précise. Les scénarios adverses (fraudeur qui optimise ses transactions pour passer sous le seuil) **ne sont pas couverts**.

## 3.4 Non-régression : référence (baseline)

`baseline/baseline.json` a été créé **manuellement** par `scripts/create_baseline.py --confirm`. Le script refuse d'écraser une référence existante sans `--confirm`, et la CI ne la régénère jamais.

| Élément enregistré | Valeur |
|---|---|
| Version du code | commit `f2338df` |
| Modèle et paramètres | Régression logistique, C = 0,01, `class_weight="balanced"`, `max_iter=2000` |
| Bibliothèques | Python 3.12.8, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1 |
| Graine | 42 (découpage et modèle) |
| Découpage | 60 / 20 / 20 stratifié, doublons supprimés |
| Seuil | 0,97 (jeu complet), 0,21 (échantillon CI) |
| Jeu de données | 284 807 lignes, 492 fraudes, 1 081 doublons, SHA-256 `64d38533…` ; échantillon : 5 000 lignes, 492 fraudes |
| Métriques de validation (jeu complet) | précision 0,491 · rappel 0,883 · FPR 0,152 % · ROC-AUC 0,981 · PR-AUC 0,792 |
| Tolérances | rappel 0,03 · précision 0,05 · PR-AUC 0,02 · ROC-AUC 0,01 · FPR 0,001 |

**Règle de décision** (`app/regression.py`) : `reject` si une dégradation dépasse la tolérance ; `investigate` si une dégradation dépasse la moitié de la tolérance, ou si le seuil optimal tombe en bordure de grille ; sinon `accept`.

## 3.5 Rapport comparatif : référence et versions candidates

Toutes les versions sont comparées sur **le même jeu de validation** (même découpage, même graine). Le test n'est pas utilisé.

| Version | Changement | Seuil | Δ rappel | Δ précision | Δ PR-AUC | Δ ROC-AUC | Δ FPR | Décision | Motif |
|---|---|---|---|---|---|---|---|---|---|
| v1.0 rejouée | Aucun (contrôle) | 0,97 | 0 | 0 | 0 | 0 | 0 | **accept** | Reproductibilité confirmée |
| v1.1 | RL avec C = 1 | 0,97 | 0,000 | −0,009 | −0,003 | +0,001 | +0,0001 | **accept** | Tous les écarts dans la tolérance |
| v1.2 | Arbre de décision réglé | 0,99 | **−0,106** | +0,286 | **−0,037** | **−0,072** | −0,0011 | **reject** | Rappel, PR-AUC et ROC-AUC dégradés au-delà de la tolérance, malgré un gain de précision |
| v1.3 | RL sans pondération des classes | **0,01** | +0,011 | +0,037 | +0,006 | −0,003 | −0,0002 | **investigate** | Métriques meilleures, mais seuil optimal en **bordure de grille** : probabilités écrasées vers 0, réglage fragile |

**Lecture.**

* v1.2 illustre pourquoi une seule métrique ne suffit pas : la précision progresse beaucoup, mais 10 fraudes de validation supplémentaires sont manquées. Cette version est rejetée.
* v1.3 est prometteuse. Avant de l'accepter, il faut affiner la grille de seuils sous 0,01 et vérifier la calibration (anomalie A-08).
* En CI, `test_no_regression_on_sample_validation` rejoue la référence sur l'échantillon versionné à chaque push. `test_no_regression_on_full_validation` le fait sur le jeu complet lors de la campagne planifiée. La PR #1 a montré qu'une régression du prétraitement est bloquée avant la fusion.
