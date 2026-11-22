"""Production des prédictions, sauvegarde et rechargement sécurisés du pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from app import config
from app.preprocessing import DataValidationError, validate_data

MAX_MODEL_FILE_MB = 50


class ModelIntegrityError(RuntimeError):
    """Fichier de modèle absent, altéré ou non conforme."""


@dataclass
class FraudModel:
    pipeline: Pipeline
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self.pipeline, X)

    def predict(self, X) -> np.ndarray:
        return predict_fraud(self.pipeline, X, self.threshold)


def _prepare(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        validated, _ = validate_data(X, require_target=False)
        return validated[config.FEATURE_COLUMNS]
    arr = np.asarray(X, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(config.FEATURE_COLUMNS):
        raise DataValidationError(
            f"{len(config.FEATURE_COLUMNS)} variables attendues, forme reçue {arr.shape}"
        )
    validated, _ = validate_data(
        pd.DataFrame(arr, columns=config.FEATURE_COLUMNS), require_target=False
    )
    return validated


def predict_proba(model: Pipeline, X) -> np.ndarray:
    """Probabilité de fraude (classe 1), garantie finie et dans [0, 1]."""
    proba = model.predict_proba(_prepare(X))[:, 1]
    if not np.isfinite(proba).all():  # garde-fou de contrat de sortie
        raise RuntimeError("Le modèle a produit une probabilité non finie")
    return proba


def predict_fraud(model: Pipeline, X, threshold: float = 0.5) -> np.ndarray:
    """Classe prédite (0 = normale, 1 = fraude) selon le seuil métier."""
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Seuil invalide : {threshold} (attendu dans ]0, 1[)")
    return (predict_proba(model, X) >= threshold).astype(int)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model: FraudModel, path) -> Path:
    """Sauvegarde joblib + fichier ``.meta.json`` (empreinte, versions, seuil)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": model.pipeline, "threshold": model.threshold, "metadata": model.metadata}, path
    )
    meta = {
        **model.metadata,
        "threshold": model.threshold,
        "sha256": _sha256(path),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "features": config.FEATURE_COLUMNS,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_model(path, verify_hash: bool = True) -> FraudModel:
    """Recharge un pipeline après contrôles (extension, taille, empreinte).

    joblib/pickle peut exécuter du code arbitraire : on ne charge qu'un fichier
    dont l'empreinte SHA-256 correspond à celle enregistrée à la sauvegarde.
    """
    path = Path(path)
    if path.suffix != ".joblib":
        raise ModelIntegrityError(f"Extension de modèle non autorisée : {path.suffix}")
    if not path.exists():
        raise ModelIntegrityError(f"Modèle introuvable : {path}")
    if path.stat().st_size > MAX_MODEL_FILE_MB * 1024 * 1024:
        raise ModelIntegrityError("Fichier de modèle anormalement volumineux")
    if verify_hash:
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise ModelIntegrityError("Métadonnées (empreinte) absentes")
        expected = json.loads(meta_path.read_text()).get("sha256")
        if expected != _sha256(path):
            raise ModelIntegrityError("Empreinte SHA-256 différente : fichier altéré")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("pipeline"), Pipeline):
        raise ModelIntegrityError("Contenu du fichier de modèle inattendu")
    return FraudModel(bundle["pipeline"], float(bundle["threshold"]), bundle.get("metadata", {}))
