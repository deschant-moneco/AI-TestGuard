"""Tests unitaires : entraînement, prédiction, seuil et métriques (T04, T08-T10)."""

import numpy as np
import pytest

from app import config
from app.evaluation import bootstrap_ci, check_acceptance, evaluate_model
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import DataValidationError
from app.training import build_pipeline, expected_cost, select_threshold, train_model
from tests.conftest import make_synthetic

# ------------------------------------------------------------------ train_model


def test_train_model_returns_fitted_pipeline_with_expected_steps(synthetic_df):
    X, y = synthetic_df[config.FEATURE_COLUMNS], synthetic_df["Class"]
    pipe = train_model(X, y)
    assert [name for name, _ in pipe.steps] == ["features", "imputer", "scaler", "model"]
    assert pipe["scaler"].n_features_in_ == 31


def test_train_model_learns_signal(synthetic_df):
    X, y = synthetic_df[config.FEATURE_COLUMNS], synthetic_df["Class"]
    pipe = train_model(X, y)
    assert evaluate_model(y, pipe.predict(X), pipe.predict_proba(X)[:, 1])["roc_auc"] > 0.9


def test_train_model_rejects_single_class(synthetic_df):
    df = synthetic_df[synthetic_df["Class"] == 0]
    with pytest.raises(DataValidationError, match="deux classes"):
        train_model(df[config.FEATURE_COLUMNS], df["Class"])


def test_train_model_rejects_inconsistent_dimensions(synthetic_df):
    with pytest.raises(DataValidationError, match="incohérentes"):
        train_model(synthetic_df[config.FEATURE_COLUMNS], synthetic_df["Class"][:-1])


def test_train_model_rejects_empty(synthetic_df):
    with pytest.raises(DataValidationError, match="vide"):
        train_model(synthetic_df[config.FEATURE_COLUMNS].iloc[:0], [])


def test_unknown_model_name():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("xgboost")


@pytest.mark.parametrize("name", ["logreg", "tree"])
def test_training_is_deterministic(synthetic_df, name):
    X, y = synthetic_df[config.FEATURE_COLUMNS], synthetic_df["Class"]
    p1 = train_model(X, y, name).predict_proba(X)[:, 1]
    p2 = train_model(X, y, name).predict_proba(X)[:, 1]
    np.testing.assert_allclose(p1, p2, atol=config.PREDICTION_ATOL)


# --------------------------------------------------------------- predict (T04)


def test_prediction_probabilities_are_valid(model, X_test):
    probabilities = predict_proba(model.pipeline, X_test)
    assert probabilities.shape == (len(X_test),)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_fraud_respects_threshold(model, X_test):
    proba = predict_proba(model.pipeline, X_test)
    for t in (0.1, 0.5, 0.9):
        np.testing.assert_array_equal(
            predict_fraud(model.pipeline, X_test, t), (proba >= t).astype(int)
        )
    # monotonie : seuil plus élevé => pas plus d'alertes
    assert (
        predict_fraud(model.pipeline, X_test, 0.9).sum()
        <= predict_fraud(model.pipeline, X_test, 0.1).sum()
    )


@pytest.mark.parametrize("t", [0.0, 1.0, -0.2, 1.5])
def test_predict_fraud_rejects_invalid_threshold(model, X_test, t):
    with pytest.raises(ValueError, match="Seuil"):
        predict_fraud(model.pipeline, X_test, t)


def test_predict_accepts_numpy_single_row(model, X_test):
    row = X_test.iloc[0].to_numpy()
    assert predict_proba(model.pipeline, row).shape == (1,)


def test_predict_rejects_wrong_number_of_features(model):
    with pytest.raises(DataValidationError, match="30 variables"):
        predict_proba(model.pipeline, np.zeros((2, 29)))


def test_predict_rejects_missing_column(model, X_test):
    with pytest.raises(DataValidationError, match="Amount"):
        predict_proba(model.pipeline, X_test.drop(columns=["Amount"]))


def test_predict_ignores_column_order(model, X_test):
    shuffled = X_test[list(reversed(config.FEATURE_COLUMNS))]
    np.testing.assert_allclose(
        predict_proba(model.pipeline, shuffled), predict_proba(model.pipeline, X_test)
    )


# ------------------------------------------------------------- evaluate_model


def test_evaluate_model_known_confusion_matrix():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    y_proba = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.05, 0.3]
    m = evaluate_model(y_true, y_pred, y_proba)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6)
    assert m["fnr"] == pytest.approx(0.25)
    assert m["roc_auc"] == pytest.approx(22 / 24)
    assert 0 < m["pr_auc"] <= 1


def test_accuracy_is_misleading_on_imbalanced_data():
    """Un modèle qui ne détecte AUCUNE fraude obtient 99,8 % d'accuracy."""
    y_true = np.array([1] * 2 + [0] * 998)
    m = evaluate_model(y_true, np.zeros(1000, dtype=int), np.zeros(1000))
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0.0 and m["precision"] == 0.0
    assert m["fnr"] == 1.0
    assert not check_acceptance(m)["accepted"]


def test_auc_undefined_when_single_class():
    m = evaluate_model([0, 0, 0], [0, 0, 1], [0.1, 0.2, 0.9])
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["fpr"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "y_true,y_pred,y_proba",
    [
        ([0, 1], [0], None),
        ([0, 2], [0, 1], None),
        ([0, 1], [0, 1], [0.1, 1.2]),
        ([0, 1], [0, 1], [0.1, np.nan]),
        ([], [], None),
    ],
)
def test_evaluate_model_rejects_invalid_inputs(y_true, y_pred, y_proba):
    with pytest.raises(ValueError):
        evaluate_model(y_true, y_pred, y_proba)


def test_check_acceptance_rules():
    good = {"recall": 0.9, "precision": 0.8, "fpr": 0.001, "pr_auc": 0.85, "roc_auc": 0.97}
    assert check_acceptance(good)["accepted"]
    bad = {**good, "fpr": 0.01}
    res = check_acceptance(bad)
    assert not res["accepted"] and not res["fpr"]["passed"] and res["recall"]["passed"]


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.8, y, 1 - y)
    lo, hi = bootstrap_ci(y, pred, "recall", n_boot=200)
    point = evaluate_model(y, pred)["recall"]
    assert lo <= point <= hi and hi - lo < 0.2


# ----------------------------------------------------------------- seuil / coût


def test_expected_cost():
    assert expected_cost([1, 1, 0, 0], [0, 1, 1, 0], c_fn=100, c_fp=5) == 105


def test_select_threshold_minimizes_cost():
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.linspace(0, 0.4, 90), np.linspace(0.6, 0.95, 10)])
    t, table = select_threshold(y, proba)
    assert 0.4 < t <= 0.6  # sépare parfaitement les classes
    assert table.loc[table.threshold == t, "cost"].item() == 0
    assert table["cost"].min() == 0


def test_select_threshold_favours_recall_when_fn_expensive():
    df = make_synthetic(2000, seed=3)
    pipe = train_model(df[config.FEATURE_COLUMNS], df["Class"])
    p = pipe.predict_proba(df[config.FEATURE_COLUMNS])[:, 1]
    t_cheap_fn, _ = select_threshold(df["Class"], p, c_fn=1, c_fp=5)
    t_costly_fn, _ = select_threshold(df["Class"], p, c_fn=500, c_fp=5)
    assert t_costly_fn <= t_cheap_fn
