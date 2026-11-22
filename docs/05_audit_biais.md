# 5. Note d'audit des biais et des limites des données

*Activité 7. Pour reproduire : `python scripts/bias_audit.py`. Chiffres bruts : `reports/bias_audit.json`, `reports/bias_audit.csv`.*

## 5.1 Limite essentielle : l'équité envers des personnes ne peut pas être évaluée

Le jeu *Credit Card Fraud* ne contient **aucun attribut de groupe** : ni âge, ni sexe, ni nationalité, ni localisation, ni type de commerçant, ni segment client. On a seulement :

* `Time` : secondes écoulées depuis la première transaction, sur 2 jours de septembre 2013 ;
* `Amount` : montant ;
* `V1`–`V28` : composantes d'une ACP appliquée à des variables d'origine confidentielles.

Par conséquent :

1. **Aucune conclusion d'équité** (rappel, FPR ou précision par groupe démographique) n'est possible avec ces données. Affirmer que le modèle est « équitable » serait infondé.
2. **On ne doit pas inventer d'attributs démographiques** à partir de V1–V28. Ces composantes mélangent des variables inconnues : leur donner un sens (« V3 = âge ») serait arbitraire et trompeur. Inférer un attribut sensible serait en outre problématique au regard du RGPD (art. 9). Nous ne l'avons pas fait.
3. Le risque de biais existe pourtant. Les variables d'origine peuvent contenir des proxys de groupes protégés (pays de la carte, type de commerçant, heure liée au mode de vie). Et un faux positif (carte bloquée) peut toucher de façon disproportionnée certaines populations, par exemple les clients qui voyagent ou ceux dont les achats sont atypiques.

## 5.2 Ce que l'on peut analyser : disparités par segment **opérationnel**

Pour repérer des zones de moindre fiabilité, on compare les performances du modèle retenu (RL, seuil 0,97) sur le **jeu de test** selon deux variables observables et non sensibles. Ce sont des segments de transactions, pas des groupes de personnes. Intervalles de confiance à 95 % : méthode de Wilson, adaptée aux petits effectifs.

| Segment | Transactions | Fraudes | Rappel [IC95] | Précision [IC95] | FPR [IC95] |
|---|---|---|---|---|---|
| **Global** | 56 746 | 95 | 0,800 [0,71 ; 0,87] | 0,494 [0,42 ; 0,57] | 0,138 % [0,11 ; 0,17] |
| Montant < 10 € | 19 523 | 47 | 0,745 [0,60 ; 0,85] | 0,625 [0,49 ; 0,74] | 0,108 % [0,07 ; 0,16] |
| Montant 10–100 € | 25 849 | 22 | 1,000 [0,85 ; 1,00] | 0,468 [0,33 ; 0,61] | 0,097 % [0,07 ; 0,14] |
| Montant 100–500 € | 9 473 | 14 | 0,786 [0,52 ; 0,92] | 0,367 [0,22 ; 0,54] | 0,201 % [0,13 ; 0,31] |
| Montant ≥ 500 € | 1 901 | 12 | 0,667 [0,39 ; 0,86] | 0,381 [0,21 ; 0,59] | **0,688 %** [0,40 ; 1,17] |
| Nuit (0 h–6 h) | 4 892 | 27 | 0,889 [0,72 ; 0,96] | 0,649 [0,49 ; 0,78] | 0,267 % [0,16 ; 0,46] |
| Jour (6 h–24 h) | 51 854 | 68 | 0,765 [0,65 ; 0,85] | 0,444 [0,36 ; 0,53] | 0,126 % [0,10 ; 0,16] |

*L'« heure » est déduite de `Time` modulo 24 h. L'origine réelle de l'horloge n'est pas documentée : « nuit » désigne la tranche 0–6 h de cette horloge relative.*

### Constats

* **Montants ≥ 500 € : FPR environ 5 fois plus élevé** (0,69 % contre 0,14 % au global). L'intervalle de confiance [0,40 % ; 1,17 %] ne recoupe pas celui du global ([0,11 % ; 0,17 %]) : l'écart est **significatif**. Les clients qui font de gros achats légitimes sont plus souvent bloqués à tort. C'est cohérent avec le scénario de robustesse 3b (le score augmente avec le montant). Dans ce segment, le FPR dépasse même le critère global de 0,5 %.
* **Rappel** : de 0,67 à 1,00 selon les segments, mais avec seulement 12 à 47 fraudes par segment, **aucun écart n'est statistiquement établi** (tous les intervalles se recoupent).
* **Jour et nuit** : la nuit, le rappel et la précision sont meilleurs, mais le FPR est 2 fois plus élevé. Les intervalles de FPR se recoupent à peine. L'écart est à surveiller, il n'est pas démontré.

## 5.3 Données complémentaires nécessaires

Pour un véritable audit d'équité, il faudrait :

| Donnée | Usage | Condition |
|---|---|---|
| Segment client (âge par tranche, ancienneté), pays de résidence | Rappel, FPR et précision par groupe, écarts et intervalles de confiance | Base légale RGPD, finalité d'audit documentée, accès restreint, **non utilisées comme variables du modèle** |
| Pays du commerçant, canal (carte présente ou non, e-commerce), MCC | Détecter des proxys et des disparités opérationnelles | Données internes de l'acquéreur |
| Issue des alertes (client joint, transaction confirmée ou non) | Mesurer le coût réel des FP par segment | Journal des analystes |
| Historique long (plusieurs mois) | Stabilité temporelle des écarts | Rétention conforme |

**Outils** : *Fairlearn* (`MetricFrame` pour les métriques par groupe, `ThresholdOptimizer`) ou *AI Fairness 360*, dès qu'un jeu avec attributs de groupe légalement utilisables est disponible. Leur application ici serait factice faute de groupes : nous ne l'avons pas fait.

## 5.4 Recommandations

1. Ne **pas** présenter le modèle comme validé du point de vue de l'équité. La mention à porter est « **non évalué faute de données** ».
2. Traiter l'écart de FPR sur les montants ≥ 500 € : examen manuel plutôt que blocage, ou seuil adapté au segment s'il est validé par le métier. Ce point est intégré à l'anomalie A-04.
3. Pendant le pilote, collecter les attributs nécessaires (§ 5.3) dans un cadre conforme, puis répéter l'audit avec Fairlearn, en fixant un écart maximal acceptable de FPR entre groupes (par exemple un ratio ≤ 1,5).
4. Faire valider la démarche par le DPO et la conformité avant tout déploiement (AI Act : un système de scoring touchant des personnes physiques demande une analyse de risque ; le RGPD encadre les décisions automatisées, art. 22).
