"""Contrôle du contrat de données et préparation des variables.

Deux comportements sont distingués explicitement :

* **Rejet** (``DataValidationError``) — la donnée viole le contrat :
  colonne requise absente ou dupliquée, jeu vide, type non numérique,
  cible hors {0, 1}, montant négatif ou absurde, temps négatif.
  En mode ``strict`` (entrée API), les valeurs manquantes / infinies et les
  colonnes supplémentaires sont aussi rejetées.
* **Correction documentée** — en mode batch (non strict) :
  ``±inf`` est converti en ``NaN`` puis imputé par la **médiane apprise sur
  l'entraînement** (``SimpleImputer`` du pipeline) ; les colonnes
  supplémentaires sont ignorées. Les corrections sont tracées dans le
  ``ValidationReport`` retourné.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app import config

logger = logging.getLogger(__name__)

ENGINEERED_COLUMNS = [*config.PCA_FEATURES, "log_amount", "hour_sin", "hour_cos"]


class DataValidationError(ValueError):
    """Donnée invalide au regard du contrat : elle doit être rejetée."""


@dataclass
class ValidationReport:
    n_rows: int
    n_missing_values: int = 0
    n_infinite_values: int = 0
    dropped_extra_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def validate_data(
    df: pd.DataFrame,
    require_target: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Vérifie le schéma et les types ; retourne une copie ordonnée + un rapport.

    Lève ``DataValidationError`` pour toute violation du contrat.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Un DataFrame est attendu, reçu {type(df).__name__}")
    if df.shape[0] == 0:
        raise DataValidationError("Jeu de données vide (0 ligne)")

    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise DataValidationError(f"Colonnes dupliquées : {sorted(set(duplicated))}")

    expected = config.REQUIRED_COLUMNS if require_target else config.FEATURE_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes requises absentes : {missing}")

    extra = [c for c in df.columns if c not in config.REQUIRED_COLUMNS]
    if extra and strict:
        raise DataValidationError(f"Colonnes non autorisées : {extra}")

    report = ValidationReport(n_rows=int(df.shape[0]), dropped_extra_columns=extra)
    if extra:
        report.warnings.append(f"Colonnes supplémentaires ignorées : {extra}")

    wrong_types = [c for c in config.FEATURE_COLUMNS if not _is_numeric(df[c])]
    if wrong_types:
        raise DataValidationError(f"Types non numériques pour : {wrong_types}")

    features = df[config.FEATURE_COLUMNS].astype("float64")
    values = features.to_numpy()
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    report.n_missing_values, report.n_infinite_values = n_nan, n_inf
    if strict and (n_nan or n_inf):
        raise DataValidationError(
            f"Valeurs manquantes ({n_nan}) ou infinies ({n_inf}) interdites en mode strict"
        )
    if n_nan or n_inf:
        report.warnings.append(
            f"{n_nan} valeur(s) manquante(s) et {n_inf} infinie(s) seront imputées"
        )

    amount = features["Amount"]
    if (amount < config.AMOUNT_MIN).any():
        raise DataValidationError("Montant négatif détecté")
    if (amount[np.isfinite(amount)] > config.AMOUNT_MAX_REJECT).any():
        raise DataValidationError(
            f"Montant supérieur à la borne de rejet ({config.AMOUNT_MAX_REJECT:,.0f})"
        )
    if (features["Time"] < config.TIME_MIN).any():
        raise DataValidationError("Valeur de Time négative détectée")

    out = features.copy()
    if require_target:
        target = df[config.TARGET_COLUMN]
        if target.isna().any():
            raise DataValidationError("Valeurs manquantes dans la cible")
        if not _is_numeric(target):
            raise DataValidationError("La cible doit être numérique (0/1)")
        unexpected = set(pd.unique(target)) - config.VALID_TARGET_VALUES
        if unexpected:
            raise DataValidationError(f"Valeurs de cible inattendues : {sorted(unexpected)}")
        out[config.TARGET_COLUMN] = target.astype("int64")

    for w in report.warnings:
        logger.warning(w)
    return out, report


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transformation **sans état** (aucun paramètre appris) des variables.

    * ``Amount`` -> ``log_amount = log1p(Amount)`` (forte asymétrie) ;
    * ``Time`` (secondes depuis la 1re transaction) -> heure de la journée
      encodée de façon cyclique (``hour_sin``, ``hour_cos``) ;
    * ``V1..V28`` conservées telles quelles (composantes PCA anonymisées,
      non interprétées comme variables métier) ;
    * ``±inf`` -> ``NaN`` (imputé ensuite par le pipeline).

    Les étapes apprises (imputation, standardisation) sont dans le
    ``Pipeline`` et ne sont ajustées que sur l'entraînement.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=config.FEATURE_COLUMNS)
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes requises absentes : {missing}")

    x = df[config.FEATURE_COLUMNS].astype("float64")  # RÉGRESSION VOLONTAIRE : inf non traité
    out = x[config.PCA_FEATURES].copy()
    out["log_amount"] = np.log1p(x["Amount"].clip(lower=0))
    hour = (x["Time"] % 86_400) / 3_600
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[ENGINEERED_COLUMNS]


def load_dataset(path) -> pd.DataFrame:
    """Charge un CSV et valide son schéma. Erreurs de chargement explicites."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    if path.suffix.lower() != ".csv":
        raise DataValidationError(f"Extension non supportée : {path.suffix}")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"CSV illisible : {exc}") from exc
    validated, _ = validate_data(df, require_target=True)
    return validated


def split_data(
    df: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VALIDATION_SIZE,
    drop_duplicates: bool = True,
):
    """Sépare train / validation / test AVANT toute transformation apprise.

    * Les doublons exacts sont supprimés au préalable : sinon une même
      transaction peut se retrouver dans train ET test (fuite).
    * Stratification sur ``Class`` pour conserver le taux de fraude.
    """
    data = df.drop_duplicates() if drop_duplicates else df
    X = data[config.FEATURE_COLUMNS]
    y = data[config.TARGET_COLUMN]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
