"""Activité 10 — fonctions de surveillance (PSI, taux d'alertes)."""

import numpy as np
import pandas as pd
import pytest

from app.monitoring import PSI_ALERT, alert_rate_check, drift_report, psi


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)


def test_psi_small_for_same_law_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref, same, shifted = rng.normal(size=5000), rng.normal(size=5000), rng.normal(1, 1, 5000)
    assert psi(ref, same) < 0.05
    assert psi(ref, shifted) > PSI_ALERT


def test_psi_ignores_non_finite_and_rejects_empty():
    assert psi([1, 2, 3, np.nan, np.inf] * 10, [1, 2, 3] * 10) == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError):
        psi([], [1, 2])


def test_drift_report_flags_only_shifted_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(2, 1, 3000)})
    rep = drift_report(ref, cur)
    assert rep.iloc[0]["feature"] == "b" and rep.iloc[0]["status"] == "alerte"
    assert rep.set_index("feature").loc["a", "status"] == "ok"


@pytest.mark.parametrize("rate,alert", [(0.003, False), (0.0005, True), (0.02, True)])
def test_alert_rate_check(rate, alert):
    preds = np.zeros(10_000)
    preds[: int(rate * 10_000)] = 1
    assert alert_rate_check(preds, expected_rate=0.003)["alert"] is alert
