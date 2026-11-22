# Comparaison des modèles — jeu de test (généré automatiquement)

Jeu : 284807 lignes, 492 fraudes. Modèle retenu (coût de validation) : **Régression logistique**.

| Modèle | Seuil | Précision | Rappel | F1 | FPR | FNR | ROC-AUC | PR-AUC | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Référence naïve (toujours normale) | - | 0.0000 | 0.0000 | 0.0000 | 0.00 % | 100.00 % | 0.5000 | 0.0017 | 99.83 % | 0 | 0 | 95 | 56651 |
| Régression logistique | 0.97 | 0.4935 | 0.8000 | 0.6104 | 0.14 % | 20.00 % | 0.9621 | 0.6783 | 99.83 % | 76 | 78 | 19 | 56573 |
| Arbre de décision | 0.99 | 0.7816 | 0.7158 | 0.7473 | 0.03 % | 28.42 % | 0.8938 | 0.6899 | 99.92 % | 68 | 19 | 27 | 56632 |
