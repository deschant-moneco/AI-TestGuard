"""Activité 8 — garde-fous de performance (version rapide pour la CI).

Les mesures détaillées (P95, débit, mémoire, protocole) sont produites par
``scripts/performance_report.py``. Ici on vérifie des bornes larges, robustes
à la variabilité des runners GitHub (marqueur ``perf``).
"""

import time

import numpy as np
import pytest

from app import config
from app.prediction import save_model

pytestmark = pytest.mark.perf


def test_single_prediction_latency_p95(model, X_test):
    row = X_test.head(1)
    for _ in range(20):  # préchauffage
        model.predict_proba(row)
    timings = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(row)
        timings.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(timings, 95))
    # borne CI = 5x le seuil d'acceptation (runners partagés, non dédiés)
    assert p95 < 5 * config.PERF_THRESHOLDS["single_latency_p95_ms"], p95


def test_batch_throughput(model, X_test):
    X = X_test.sample(10_000, replace=True, random_state=0)
    t0 = time.perf_counter()
    model.predict_proba(X)
    throughput = len(X) / (time.perf_counter() - t0)
    assert throughput > config.PERF_THRESHOLDS["batch_throughput_min_per_s"] / 5, throughput


def test_model_file_size(tmp_path, model):
    path = save_model(model, tmp_path / "m.joblib")
    assert path.stat().st_size / 1e6 < config.PERF_THRESHOLDS["model_size_max_mb"]
