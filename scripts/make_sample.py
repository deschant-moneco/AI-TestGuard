"""Construit l'échantillon versionné data/sample_creditcard.csv (utilisé en CI).

Toutes les fraudes (492) + 4 508 transactions normales tirées au hasard
(graine fixe). L'échantillon est ENRICHI en fraudes (~9,8 % au lieu de
0,17 %) : il sert à tester le pipeline et la non-régression, pas à estimer
la performance réelle (voir data/README.md).
"""

import pandas as pd

from app import config

N_TOTAL = 5000

df = pd.read_csv(config.FULL_DATASET)
fraud = df[df[config.TARGET_COLUMN] == 1]
normal = df[df[config.TARGET_COLUMN] == 0].sample(
    N_TOTAL - len(fraud), random_state=config.RANDOM_SEED
)
sample = pd.concat([fraud, normal]).sort_values("Time").reset_index(drop=True)
sample.to_csv(config.SAMPLE_DATASET, index=False)
print(f"{len(sample)} lignes, {int(sample.Class.sum())} fraudes -> {config.SAMPLE_DATASET}")
