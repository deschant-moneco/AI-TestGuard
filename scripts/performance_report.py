"""Activité 8 — protocole de mesure de performance.

Protocole :
* matériel/OS relevés automatiquement (platform, psutil) ;
* 50 appels de préchauffage, puis N=2000 répétitions pour la latence unitaire ;
* latence « modèle » (DataFrame 1 ligne -> validation + predict_proba) distinguée
  de la latence « requête API » (HTTP local + JSON + validation stricte) ;
* débit mesuré par lots de 1, 100, 1 000, 10 000 et 56 746 lignes (5 répétitions,
  médiane retenue) ;
* mémoire : RSS (psutil) avant/après chargement du modèle et pendant le scoring
  du jeu de test complet ; pic Python via tracemalloc.
Toutes les durées : time.perf_counter.
"""

import json
import os
import platform
import time
import tracemalloc
import urllib.request

import numpy as np
import psutil

from app import config
from app.api import serve
from app.prediction import load_model
from app.preprocessing import load_dataset, split_data

N_WARMUP, N_REP = 50, 2000


def pct(values):
    a = np.asarray(values)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
        "n": int(len(a)),
    }


def main():
    proc = psutil.Process(os.getpid())
    df = load_dataset(config.FULL_DATASET)
    X_test = split_data(df)[2]
    del df
    rss_before = proc.memory_info().rss / 1e6
    path = config.MODELS_DIR / "fraud_model.joblib"
    model = load_model(path)
    rss_after_load = proc.memory_info().rss / 1e6

    row = X_test.head(1)
    for _ in range(N_WARMUP):
        model.predict_proba(row)
    single = []
    for i in range(N_REP):
        r = X_test.iloc[[i % len(X_test)]]
        t0 = time.perf_counter()
        model.predict_proba(r)
        single.append((time.perf_counter() - t0) * 1000)

    # Latence du modèle brut (sans validation applicative) pour isoler le coût
    raw = []
    for i in range(N_REP):
        r = X_test.iloc[[i % len(X_test)]]
        t0 = time.perf_counter()
        model.pipeline.predict_proba(r)
        raw.append((time.perf_counter() - t0) * 1000)

    # Requête API HTTP locale (sérialisation + validation stricte + scoring)
    server, port = serve(model)
    url = f"http://127.0.0.1:{port}/predict"
    api = []
    try:
        for i in range(N_WARMUP + 1000):
            tx = {k: float(v) for k, v in X_test.iloc[i % len(X_test)].items()}
            data = json.dumps({"transactions": [tx]}).encode()
            req = urllib.request.Request(
                url, data=data, method="POST", headers={"Content-Type": "application/json"}
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            if i >= N_WARMUP:
                api.append((time.perf_counter() - t0) * 1000)
    finally:
        server.shutdown()

    throughput = {}
    for size in (1, 100, 1_000, 10_000, len(X_test)):
        batch = X_test.sample(size, replace=size > len(X_test), random_state=0)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.predict_proba(batch)
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        throughput[str(size)] = {"median_s": med, "pred_per_s": size / med}

    tracemalloc.start()
    rss_peak_before = proc.memory_info().rss / 1e6
    model.predict_proba(X_test)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_scoring = proc.memory_info().rss / 1e6

    out = {
        "environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "protocol": {
            "warmup": N_WARMUP,
            "repetitions_single": N_REP,
            "repetitions_api": 1000,
            "batch_repetitions": 5,
        },
        "single_prediction_with_validation": pct(single),
        "single_prediction_raw_pipeline": pct(raw),
        "api_http_request": pct(api),
        "throughput": throughput,
        "model_file_kb": round(path.stat().st_size / 1024, 1),
        "memory": {
            "rss_before_load_mb": rss_before,
            "rss_after_load_mb": rss_after_load,
            "model_load_delta_mb": rss_after_load - rss_before,
            "rss_scoring_delta_mb": rss_after_scoring - rss_peak_before,
            "tracemalloc_peak_scoring_test_mb": peak / 1e6,
        },
        "thresholds": config.PERF_THRESHOLDS,
    }
    (config.REPORTS_DIR / "performance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
