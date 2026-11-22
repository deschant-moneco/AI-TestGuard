"""Activité 4 — tests d'intégration de bout en bout et prévention des fuites."""

import numpy as np
import pandas as pd
import pytest

from app import config
from app.evaluation import evaluate_model
from app.experiment import run_experiment
from app.prediction import ModelIntegrityError, load_model, save_model
from app.preprocessing import DataValidationError, load_dataset, split_data


def test_end_to_end_pipeline(tmp_path, sample_df):
    """charger -> valider -> split -> ajuster -> seuil (val) -> sauver -> recharger
    -> prédire sur test -> rapport de métriques."""
    csv = tmp_path / "data.csv"
    sample_df.to_csv(csv, index=False)
    df = load_dataset(csv)  # étapes 1-2
    exp = run_experiment(df, "logreg", params={"C": 0.01})  # étapes 3-5
    path = save_model(exp.model, tmp_path / "model.joblib")  # 6
    reloaded = load_model(path)  # 7
    X_test, y_test = exp.splits[2], exp.splits[5]
    proba = reloaded.predict_proba(X_test)  # 8
    report = evaluate_model(y_test, reloaded.predict(X_test), proba)  # 9
    assert report["n"] == len(X_test)
    assert report["tp"] + report["fn"] == int(y_test.sum())
    assert report["roc_auc"] > 0.9  # garde-fou grossier sur l'échantillon


def test_no_non_finite_values_after_preprocessing(model, X_test):
    transformed = model.pipeline[:-1].transform(X_test)
    assert np.isfinite(transformed).all()


def test_dimensions_consistent_in_and_out(model, X_test):
    transformed = model.pipeline[:-1].transform(X_test)
    assert transformed.shape == (len(X_test), 31)
    assert model.predict_proba(X_test).shape == (len(X_test),)
    assert model.predict(X_test).shape == (len(X_test),)


def test_predictions_identical_after_save_and_reload(tmp_path, model, X_test):
    path = save_model(model, tmp_path / "m.joblib")
    reloaded = load_model(path)
    np.testing.assert_allclose(
        reloaded.predict_proba(X_test),
        model.predict_proba(X_test),
        atol=config.PREDICTION_ATOL,
        rtol=0,
    )
    np.testing.assert_array_equal(reloaded.predict(X_test), model.predict(X_test))
    assert reloaded.threshold == model.threshold


def test_scaler_and_imputer_fitted_on_train_only(trained):
    """Les statistiques apprises doivent être celles de l'entraînement, pas du test."""
    X_train, X_val, X_test = trained.splits[:3]
    pipe = trained.model.pipeline
    feats_train = pipe["features"].transform(X_train)
    feats_test = pipe["features"].transform(X_test)
    np.testing.assert_allclose(pipe["scaler"].mean_, feats_train.mean().to_numpy(), atol=1e-8)
    np.testing.assert_allclose(
        pipe["imputer"].statistics_, feats_train.median().to_numpy(), atol=1e-8
    )
    assert not np.allclose(pipe["scaler"].mean_, feats_test.mean().to_numpy(), atol=1e-6)
    assert pipe["scaler"].n_samples_seen_ == len(X_train)


def test_test_set_never_used_for_threshold_or_tuning(sample_df):
    """Modifier les étiquettes du test ne doit changer ni le seuil ni le modèle."""
    splits = split_data(sample_df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits
    a = run_experiment(sample_df, "logreg", splits=splits)
    shuffled = (
        X_tr,
        X_va,
        X_te.sample(frac=1, random_state=1),
        y_tr,
        y_va,
        pd.Series(np.random.default_rng(0).permutation(y_te.to_numpy()), index=y_te.index),
    )
    b = run_experiment(sample_df, "logreg", splits=shuffled)
    assert a.threshold == b.threshold
    assert a.params == b.params
    np.testing.assert_allclose(a.model.predict_proba(X_va), b.model.predict_proba(X_va))


def test_splits_are_disjoint_without_duplicate_rows(trained):
    X_tr, X_va, X_te = trained.splits[:3]
    assert not set(X_tr.index) & set(X_te.index)
    assert not set(X_va.index) & set(X_te.index)
    assert pd.concat([X_tr, X_te]).duplicated().sum() == 0


def test_load_model_rejects_tampered_file(tmp_path, model):
    path = save_model(model, tmp_path / "m.joblib")
    with open(path, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ModelIntegrityError, match="altéré"):
        load_model(path)


def test_load_model_rejects_bad_extension_missing_file_and_metadata(tmp_path, model):
    with pytest.raises(ModelIntegrityError, match="Extension"):
        load_model(tmp_path / "m.pkl")
    with pytest.raises(ModelIntegrityError, match="introuvable"):
        load_model(tmp_path / "absent.joblib")
    path = save_model(model, tmp_path / "m.joblib")
    path.with_suffix(".meta.json").unlink()
    with pytest.raises(ModelIntegrityError, match="Métadonnées"):
        load_model(path)


def test_saved_metadata_contains_versions(tmp_path, model):
    import json

    path = save_model(model, tmp_path / "m.joblib")
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    for key in ("sha256", "sklearn_version", "python_version", "threshold", "features"):
        assert key in meta


def test_invalid_input_rejected_by_reloaded_pipeline(tmp_path, model, X_test):
    reloaded = load_model(save_model(model, tmp_path / "m.joblib"))
    bad = X_test.head(3).assign(Amount="cent")
    with pytest.raises(DataValidationError):
        reloaded.predict_proba(bad)
