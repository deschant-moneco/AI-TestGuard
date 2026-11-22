"""Point d'entrée de scoring (prototype) : validation stricte + journalisation sûre.

``handle_request`` est indépendant du transport HTTP (testable unitairement).
``serve`` expose la même logique via ``http.server`` (bibliothèque standard),
uniquement pour mesurer la latence d'une requête HTTP complète en local.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from app import config
from app.prediction import FraudModel
from app.preprocessing import DataValidationError, validate_data

logger = logging.getLogger("fraudcheck.api")

MAX_BODY_BYTES = 16 * 1024
MAX_TRANSACTIONS_PER_REQUEST = 100


def _transaction_id(tx: dict) -> str:
    """Identifiant pseudonyme (hash tronqué) : aucune valeur brute en journal."""
    raw = json.dumps(tx, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def parse_payload(body: str | bytes) -> list[dict]:
    if isinstance(body, bytes):
        if len(body) > MAX_BODY_BYTES:
            raise DataValidationError("Requête trop volumineuse")
        body = body.decode("utf-8")
    elif len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise DataValidationError("Requête trop volumineuse")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DataValidationError("JSON invalide") from exc
    if not isinstance(payload, dict) or set(payload) != {"transactions"}:
        raise DataValidationError('Le corps doit être {"transactions": [...]}')
    txs = payload["transactions"]
    if not isinstance(txs, list) or not txs:
        raise DataValidationError("'transactions' doit être une liste non vide")
    if len(txs) > MAX_TRANSACTIONS_PER_REQUEST:
        raise DataValidationError("Trop de transactions dans la requête")
    expected = set(config.FEATURE_COLUMNS)
    for i, tx in enumerate(txs):
        if not isinstance(tx, dict):
            raise DataValidationError(f"Transaction {i} : objet JSON attendu")
        missing, extra = expected - set(tx), set(tx) - expected
        if missing:
            raise DataValidationError(f"Transaction {i} : champs manquants {sorted(missing)}")
        if extra:
            raise DataValidationError(f"Transaction {i} : champs non autorisés {sorted(extra)}")
        for k, v in tx.items():
            # bool est une sous-classe de int en Python : on le refuse explicitement
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DataValidationError(f"Transaction {i} : '{k}' doit être numérique")
            if not math.isfinite(v):
                raise DataValidationError(f"Transaction {i} : '{k}' non fini")
    return txs


def handle_request(model: FraudModel, body: str | bytes) -> tuple[int, dict]:
    """Retourne (code HTTP, réponse). Ne lève pas : les erreurs deviennent 400/500."""
    try:
        txs = parse_payload(body)
        df, _ = validate_data(pd.DataFrame(txs), require_target=False, strict=True)
        proba = model.predict_proba(df)
        preds = (proba >= model.threshold).astype(int)
    except DataValidationError as exc:
        logger.info("requête rejetée: %s", exc)
        return 400, {"error": str(exc)}
    except Exception:  # pragma: no cover - garde-fou
        logger.exception("erreur interne de scoring")
        return 500, {"error": "Erreur interne"}
    results = []
    for tx, p, c in zip(txs, proba, preds, strict=True):
        tid = _transaction_id(tx)
        logger.info("scored tx=%s fraud=%d", tid, int(c))  # pas de montant ni de V*
        results.append({"id": tid, "fraud_probability": round(float(p), 6), "is_fraud": bool(c)})
    return 200, {"threshold": model.threshold, "results": results}


def make_handler(model: FraudModel):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (API http.server)
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "Requête trop volumineuse"})
                return
            status, resp = handle_request(model, self.rfile.read(length))
            self._send(status, resp)

        def _send(self, status, resp):
            data = json.dumps(resp).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # pas de journalisation des requêtes brutes
            return

    return Handler


def serve(model: FraudModel, host: str = "127.0.0.1", port: int = 0):
    """Démarre un serveur local dans un thread ; retourne (serveur, port)."""
    server = ThreadingHTTPServer((host, port), make_handler(model))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]
