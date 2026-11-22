# Données

| Fichier | Versionné | Contenu |
|---|---|---|
| `creditcard.csv` | **non** (150 Mo, `.gitignore`) | Jeu complet *Credit Card Fraud Detection* (ULB / Worldline), 284 807 transactions dont 492 fraudes. |
| `sample_creditcard.csv` | oui (2,6 Mo) | Échantillon de 5 000 lignes : **les 492 fraudes** + 4 508 transactions normales tirées au hasard (graine 42). Généré par `scripts/make_sample.py`. |

## Source et licence

* Kaggle : <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> — licence *Open Database License (ODbL) / DbCL v1.0*.
* Miroir public identique utilisé pour l'automatisation (sans compte Kaggle) :
  `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`
* Empreinte SHA-256 du fichier téléchargé : `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
  (vérifiée dans le workflow `full`).

```bash
curl -sSL -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
shasum -a 256 data/creditcard.csv
```

## Précautions

* L'échantillon est **enrichi en fraudes** (9,8 % au lieu de 0,17 %) : il sert à
  tester le pipeline et la non-régression en CI, **pas** à estimer la
  performance réelle (précision et FPR y sont mécaniquement très différents).
* `V1`–`V28` sont des composantes PCA anonymisées : elles ne sont pas
  interprétées comme des variables métier.
* Le jeu contient 1 081 doublons exacts, supprimés avant le découpage
  train / validation / test pour éviter les fuites.
* Les données datent de septembre 2013 (2 jours, porteurs européens) : elles
  ne prouvent rien, seules, sur les transactions bancaires actuelles.
