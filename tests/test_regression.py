"""Activité 6 — reproductibilité (T06) et non-régression contre une baseline figée."""

import json

import numpy as np
import pytest

from app import config
from app.experiment import run_experiment
from app.preprocessing import load_dataset
from app.regression import compare_to_baseline

BASELINE = json.loads((config.BASELINE_DIR / "baseline.json").read_text())


def test_baseline_records_required_information():
    for key in ("environment", "params", "seed", "tolerance", "sample", "model", "costs"):
        assert key in BASELINE
    env = BASELINE["environment"]
    for key in ("code_version", "scikit_learn", "numpy", "pandas", "python"):
        assert key in env
    snap = BASELINE["sample"]
    assert {"threshold", "validation_metrics", "dataset"} <= set(snap)


def test_sample_dataset_unchanged_since_baseline(sample_df):
    from app.experiment import dataset_fingerprint

    assert dataset_fingerprint(sample_df) == BASELINE["sample"]["dataset"]


def test_two_runs_are_identical(sample_df):
    a = run_experiment(sample_df, BASELINE["model"], params=BASELINE["params"])
    b = run_experiment(sample_df, BASELINE["model"], params=BASELINE["params"])
    X_val = a.splits[1]
    np.testing.assert_allclose(
        a.model.predict_proba(X_val),
        b.model.predict_proba(X_val),
        atol=config.PREDICTION_ATOL,
        rtol=0,
    )
    assert a.threshold == b.threshold


def test_no_regression_on_sample_validation(trained):
    snap = BASELINE["sample"]
    report = compare_to_baseline(
        snap["validation_metrics"], trained.val_metrics, BASELINE["tolerance"]
    )
    failing = {m: r for m, r in report["metrics"].items() if r["status"] == "régression"}
    assert report["decision"] != "reject", f"Régression détectée : {failing}"
    assert abs(trained.threshold - snap["threshold"]) <= 0.05


@pytest.mark.slow
@pytest.mark.skipif(not config.FULL_DATASET.exists(), reason="jeu complet absent")
def test_no_regression_on_full_validation():
    snap = BASELINE["full"]
    exp = run_experiment(
        load_dataset(config.FULL_DATASET), BASELINE["model"], params=BASELINE["params"]
    )
    report = compare_to_baseline(snap["validation_metrics"], exp.val_metrics, BASELINE["tolerance"])
    assert report["decision"] != "reject", report


# ------------------------------------------------------- logique de comparaison


BASE = {"recall": 0.85, "precision": 0.60, "pr_auc": 0.80, "roc_auc": 0.97, "fpr": 0.001}


def test_compare_accepts_identical_metrics():
    assert compare_to_baseline(BASE, BASE)["decision"] == "accept"


def test_compare_rejects_recall_drop_beyond_tolerance():
    cand = {**BASE, "recall": BASE["recall"] - 0.05}
    r = compare_to_baseline(BASE, cand)
    assert r["decision"] == "reject"
    assert r["metrics"]["recall"]["status"] == "régression"


def test_compare_rejects_fpr_increase():
    r = compare_to_baseline(BASE, {**BASE, "fpr": BASE["fpr"] + 0.002})
    assert r["decision"] == "reject"


def test_compare_investigates_small_degradation_and_accepts_improvement():
    small = {**BASE, "precision": BASE["precision"] - 0.03}  # > tol/2, < tol
    assert compare_to_baseline(BASE, small)["decision"] == "investigate"
    better = {**BASE, "recall": 0.95, "pr_auc": 0.9}
    r = compare_to_baseline(BASE, better)
    assert r["decision"] == "accept"
    assert r["metrics"]["recall"]["status"] == "amélioration"
