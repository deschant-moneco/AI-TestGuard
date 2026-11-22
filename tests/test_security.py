"""Activité 8 — sécurité opérationnelle : validation API, journaux, secrets."""

import json
import logging
import re
import subprocess
import urllib.error
import urllib.request

import pytest

from app import config
from app.api import MAX_BODY_BYTES, handle_request, parse_payload, serve
from app.preprocessing import DataValidationError


def tx_from(X_test, i=0) -> dict:
    return {k: float(v) for k, v in X_test.iloc[i].items()}


def body(*txs) -> str:
    return json.dumps({"transactions": list(txs)})


def test_valid_request_is_scored(model, X_test):
    status, resp = handle_request(model, body(tx_from(X_test), tx_from(X_test, 1)))
    assert status == 200
    assert len(resp["results"]) == 2
    for r in resp["results"]:
        assert 0 <= r["fraud_probability"] <= 1
        assert isinstance(r["is_fraud"], bool)


def test_missing_field_is_rejected(model, X_test):
    tx = tx_from(X_test)
    del tx["V17"]
    status, resp = handle_request(model, body(tx))
    assert status == 400 and "V17" in resp["error"]


def test_extra_field_is_rejected(model, X_test):
    status, resp = handle_request(model, body({**tx_from(X_test), "card_number": 4111}))
    assert status == 400 and "non autorisés" in resp["error"]


@pytest.mark.parametrize("bad", ["12.5", None, True, [1], {"x": 1}])
def test_incorrect_types_are_rejected(model, X_test, bad):
    status, resp = handle_request(model, body({**tx_from(X_test), "Amount": bad}))
    assert status == 400 and "numérique" in resp["error"]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        '{"transactions": []}',
        '{"transactions": "x"}',
        '{"transactions": [1]}',
        '{"transactions": [], "admin": true}',
        '{"transactions": [{"Time": NaN}]}',
    ],
)
def test_malformed_payloads_are_rejected(model, payload):
    status, _ = handle_request(model, payload)
    assert status == 400


def test_nan_and_infinity_rejected_in_api(model, X_test):
    raw = body(tx_from(X_test)).replace(
        f'"Amount": {tx_from(X_test)["Amount"]}', '"Amount": Infinity'
    )
    assert handle_request(model, raw)[0] == 400


def test_negative_amount_rejected_in_api(model, X_test):
    status, resp = handle_request(model, body({**tx_from(X_test), "Amount": -10.0}))
    assert status == 400 and "négatif" in resp["error"]


def test_oversized_payload_and_too_many_transactions(model, X_test):
    with pytest.raises(DataValidationError, match="volumineuse"):
        parse_payload(b"x" * (MAX_BODY_BYTES + 1))
    status, _ = handle_request(model, body(*[tx_from(X_test)] * 101))
    assert status == 400


def test_logs_do_not_leak_sensitive_values(model, X_test, caplog):
    caplog.set_level(logging.DEBUG, logger="ai_testguard")
    tx = {**tx_from(X_test), "Amount": 9876.54}
    handle_request(model, body(tx))
    handle_request(model, body({**tx, "Amount": "oops"}))
    text = caplog.text
    assert "scored tx=" in text  # la journalisation existe bien
    assert "9876.54" not in text
    assert f"{tx['V1']}" not in text
    assert "Amount" not in text.replace("'Amount' doit", "")  # seul le nom du champ en erreur


def test_http_endpoint_end_to_end(model, X_test):
    server, port = serve(model)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=body(tx_from(X_test)).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
            assert "results" in json.loads(r.read())
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad, timeout=5)
        assert exc.value.code == 400
    finally:
        server.shutdown()


# ------------------------------------------------------------- secrets / dépôt

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # clé AWS
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),  # jeton GitHub
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"KGAT_[A-Za-z0-9]{16,}"),  # jeton Kaggle
]
SCANNED_SUFFIXES = {".py", ".yml", ".yaml", ".json", ".md", ".txt", ".cfg", ".toml", ".ini", ".env"}


def _repo_files():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=config.ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [config.ROOT_DIR / f for f in out.stdout.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in config.ROOT_DIR.rglob("*") if ".venv" not in p.parts]
    return [f for f in files if f.is_file()]


def test_no_secrets_in_repository():
    hits = []
    for f in _repo_files():
        if f.suffix not in SCANNED_SUFFIXES or f.name == "test_security.py":
            continue
        text = f.read_text(errors="ignore")
        hits += [f"{f.name}: {p.pattern}" for p in SECRET_PATTERNS if p.search(text)]
    assert not hits, hits


def test_sensitive_files_are_gitignored():
    ignored = (config.ROOT_DIR / ".gitignore").read_text()
    for pattern in (".env", "kaggle.json", "data/creditcard.csv", "models/"):
        assert pattern in ignored
    names = {f.name for f in _repo_files()}
    assert ".env" not in names and "kaggle.json" not in names
    assert "creditcard.csv" not in names  # jeu complet (150 Mo) non versionné
