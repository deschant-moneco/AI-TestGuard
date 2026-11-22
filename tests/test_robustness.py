"""Activité 5 — robustesse et sensibilité sur données imparfaites.

Pour chaque scénario, le comportement attendu est précisé : REJET, TRAITEMENT
(correction documentée) ou ALERTE. On mesure la sensibilité plutôt que
d'imposer une invariance universelle.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app import config
from app.monitoring import alert_rate_check, drift_report
from app.preprocessing import DataValidationError
from app.robustness import (
    covariate_shift,
    gaussian_noise_pca,
    inject_missing,
    relative_amount_noise,
    scale_amount,
    sensitivity,
    shift_time,
)


@pytest.fixture(scope="module")
def stds(trained):
    return trained.splits[0].std()


# 1. Données manquantes -> TRAITER (imputation médiane apprise) en batch,
#    REJETER en mode strict (API).
def test_missing_values_are_imputed_and_bounded_impact(model, X_test, y_test):
    Xp, n = inject_missing(X_test, 0.10, ["V4", "V14"], seed=1)
    assert n > 0
    res = sensitivity(model, X_test, Xp, y_test)
    assert not res["rejected"]
    assert res["pct_class_changes"] < 0.10  # impact limité, mesuré
    p = model.predict_proba(Xp)
    assert np.isfinite(p).all()


def test_fully_missing_row_is_still_scored(model, X_test):
    X = X_test.head(1).copy()
    X.loc[:, :] = np.nan
    assert np.isfinite(model.predict_proba(X)).all()


# 2. Valeurs infinies ou non numériques
def test_infinite_values_are_treated(model, X_test):
    X = X_test.head(20).copy()
    X.iloc[:5, 1] = np.inf
    assert np.isfinite(model.predict_proba(X)).all()


def test_non_numeric_values_are_rejected(model, X_test):
    X = X_test.head(5).astype({"V10": object})
    X.iloc[0, X.columns.get_loc("V10")] = "N/A"
    with pytest.raises(DataValidationError, match="V10"):
        model.predict_proba(X)


# 3. Montants extrêmes : <= 1 M€ traités (probabilité valide), au-delà rejetés
@pytest.mark.parametrize("amount", [0.0, 25_691.16, 100_000.0, 1_000_000.0])
def test_extreme_amounts_give_valid_probabilities(model, X_test, amount):
    X = X_test.head(10).assign(Amount=amount)
    p = model.predict_proba(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_amount_above_contract_is_rejected(model, X_test):
    with pytest.raises(DataValidationError, match="borne"):
        model.predict_proba(scale_amount(X_test.head(10), 1e7))


# 4. Perturbation limitée -> sensibilité mesurée et bornée
@pytest.mark.parametrize("eps,max_changes", [(0.01, 0.01), (0.05, 0.03), (0.10, 0.05)])
def test_small_pca_noise_has_bounded_effect(model, X_test, y_test, stds, eps, max_changes):
    res = sensitivity(model, X_test, gaussian_noise_pca(X_test, eps, stds), y_test)
    assert res["pct_class_changes"] <= max_changes
    assert abs(res["delta_recall"]) <= 0.05


def test_sensitivity_increases_with_noise_amplitude(model, X_test, y_test, stds):
    deltas = [
        sensitivity(model, X_test, gaussian_noise_pca(X_test, e, stds), y_test)[
            "mean_abs_delta_proba"
        ]
        for e in (0.01, 0.05, 0.10, 0.5)
    ]
    assert deltas == sorted(deltas)


def test_amount_relative_noise_and_time_shift(model, X_test, y_test):
    r1 = sensitivity(model, X_test, relative_amount_noise(X_test, 0.05), y_test)
    r2 = sensitivity(model, X_test, shift_time(X_test, 3600), y_test)
    assert r1["pct_class_changes"] <= 0.02
    assert r2["pct_class_changes"] <= 0.05


# 5. Distribution différente -> ALERTE (dérive détectée par le monitoring)
def test_covariate_shift_triggers_drift_alert(trained, model, X_test, stds):
    X_train = trained.splits[0]
    shifted = covariate_shift(X_test, stds)
    report = drift_report(X_train, shifted, config.FEATURE_COLUMNS)
    assert (report["status"] == "alerte").sum() >= 10
    ref_rate = float(model.predict(trained.splits[1]).mean())
    check = alert_rate_check(model.predict(shifted), ref_rate)
    # Constat (anomalie A-05) : le taux d'alertes chute d'environ 40 % mais reste
    # dans la bande ×2 ; seul le PSI détecte la dérive -> les deux signaux sont requis.
    assert abs(check["rate"] - ref_rate) / ref_rate > 0.30
    assert not check["alert"]
    strong = alert_rate_check(
        model.predict(covariate_shift(X_test, stds, k=-0.5, amount_factor=1)), ref_rate
    )
    assert strong["alert"]  # une dérive dans l'autre sens fait exploser les alertes


def test_no_drift_alert_on_same_distribution(trained, X_test):
    report = drift_report(trained.splits[0], X_test, config.FEATURE_COLUMNS)
    assert (report["status"] == "alerte").sum() == 0


# 6. Schéma invalide -> REJET
@pytest.mark.parametrize(
    "mutate",
    [
        lambda X: X.drop(columns=["V1"]),
        lambda X: X.rename(columns={"Amount": "amount"}),
        lambda X: pd.concat([X, X[["V2"]]], axis=1),
        lambda X: X.iloc[:0],
    ],
)
def test_invalid_schema_is_rejected(model, X_test, mutate):
    with pytest.raises(DataValidationError):
        model.predict_proba(mutate(X_test.head(5)))


# Propriété : pour toute entrée finie respectant le contrat, proba ∈ [0, 1]
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    v=arrays(np.float64, 28, elements=st.floats(-50, 50)),
    amount=st.floats(0, 1e6),
    time=st.floats(0, 2e5),
)
def test_property_probability_always_valid(model, v, amount, time):
    row = pd.DataFrame([[time, *v, amount]], columns=config.FEATURE_COLUMNS)
    p = model.predict_proba(row)
    assert np.isfinite(p).all() and 0 <= p[0] <= 1
