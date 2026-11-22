"""Crée la référence (baseline) de non-régression — action MANUELLE et tracée.

La baseline n'est JAMAIS recalculée automatiquement par la CI : sinon une
régression deviendrait la nouvelle référence et passerait inaperçue.
Usage : python scripts/create_baseline.py --confirm
"""

import argparse
import json
from datetime import UTC, datetime

from app import config
from app.experiment import dataset_fingerprint, environment_info, run_experiment
from app.preprocessing import load_dataset

BASELINE_MODEL = "logreg"
BASELINE_PARAMS = {"C": 0.01}
KEYS = ("precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn")


def snapshot(path):
    df = load_dataset(path)
    exp = run_experiment(df, BASELINE_MODEL, params=BASELINE_PARAMS)
    return {
        "dataset": dataset_fingerprint(df),
        "threshold": exp.threshold,
        "validation_metrics": {k: exp.val_metrics[k] for k in KEYS},
        "validation_size": int(len(exp.splits[4])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="écrase la baseline existante")
    args = ap.parse_args()
    out = config.BASELINE_DIR / "baseline.json"
    if out.exists() and not args.confirm:
        raise SystemExit("Baseline existante : relancer avec --confirm après revue humaine.")
    baseline = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "model": BASELINE_MODEL,
        "params": BASELINE_PARAMS,
        "seed": config.RANDOM_SEED,
        "split": {
            "test_size": config.TEST_SIZE,
            "val_size": config.VALIDATION_SIZE,
            "stratified": True,
            "drop_duplicates": True,
        },
        "costs": {"fn": config.COST_FALSE_NEGATIVE, "fp": config.COST_FALSE_POSITIVE},
        "tolerance": config.REGRESSION_TOLERANCE,
        "sample": snapshot(config.SAMPLE_DATASET),
    }
    if config.FULL_DATASET.exists():
        baseline["full"] = snapshot(config.FULL_DATASET)
    config.BASELINE_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
