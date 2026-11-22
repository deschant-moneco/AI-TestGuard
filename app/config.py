"""Configuration centrale : contrat de données, graines, seuils d'acceptation.

Toutes les valeurs « métier » sont définies ici, AVANT toute observation du jeu
de test, afin d'éviter de régler les critères a posteriori.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
BASELINE_DIR = ROOT_DIR / "baseline"

FULL_DATASET = DATA_DIR / "creditcard.csv"
SAMPLE_DATASET = DATA_DIR / "sample_creditcard.csv"

# --- Contrat de données -------------------------------------------------------
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *PCA_FEATURES, "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
VALID_TARGET_VALUES = {0, 1}

# Bornes « physiques » : au-delà, la donnée est rejetée (pas corrigée).
AMOUNT_MIN = 0.0
AMOUNT_MAX_REJECT = 1_000_000.0  # montant absurde pour une transaction carte
TIME_MIN = 0.0

# --- Reproductibilité ---------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # fraction du jeu complet

# --- Coûts métier utilisés pour choisir le seuil (sur validation uniquement) ---
# Hypothèses documentées dans docs/plan_de_test.md :
#   - un faux négatif coûte le montant moyen d'une fraude (~122 €) + frais de
#     traitement / litige : arrondi à 150 € ;
#   - un faux positif coûte une analyse manuelle + friction client : ~5 €.
COST_FALSE_NEGATIVE = 150.0
COST_FALSE_POSITIVE = 5.0

# --- Critères d'acceptation (fixés a priori) ---------------------------------
ACCEPTANCE_CRITERIA = {
    "recall_min": 0.75,  # au moins 3 fraudes sur 4 détectées
    "precision_min": 0.50,  # au plus 1 alerte sur 2 est une fausse alerte
    "fpr_max": 0.005,  # <= 0,5 % de clients légitimes bloqués à tort
    "pr_auc_min": 0.70,  # qualité globale sur classe rare
    "roc_auc_min": 0.90,
}

# --- Non-régression -----------------------------------------------------------
REGRESSION_TOLERANCE = {
    "recall": 0.03,
    "precision": 0.05,
    "pr_auc": 0.02,
    "roc_auc": 0.01,
    "fpr": 0.001,
}

# Tolérance numérique de reproductibilité (avant / après rechargement, 2 runs)
PREDICTION_ATOL = 1e-9

# --- Performance (cf. docs/rapport_performance.md) ----------------------------
PERF_THRESHOLDS = {
    "single_latency_p95_ms": 10.0,  # scoring en ligne d'une transaction
    "api_latency_p95_ms": 50.0,  # requête HTTP locale complète
    "batch_throughput_min_per_s": 20_000,
    "model_size_max_mb": 5.0,
    "memory_increase_max_mb": 200.0,
}
