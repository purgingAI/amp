#!/usr/bin/env python3
"""HTTP face for the monitor, so the signing laptop can reach it over the network.

The device key stays on laptop 1 and never travels. What crosses the wire is a signature over
a nonce the monitor issued — which is worth nothing to anyone who intercepts it: it is bound to
one mandate, one payee, one amount and one single-use nonce.

Three endpoints, nothing else:

  POST /pay/session    {agent_id, mandate_id, eph_pub, nonce}  -> opaque token
  POST /pay/nonce      {token}                                 -> single-use nonce
  POST /pay/authorize  {token, nonce, payee_wallet, amount, currency, timestamp,
                        signature, step_up_id?}                -> captured | requires_step_up

Deliberately absent: anything that mints a mandate, enrols a key, or moves money without a
device signature. The dangerous operations stay off the network.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import monitor, mandate_store as ms, ledger, notify, trace, step_up   # noqa: E402

MAX_BODY = 8192


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        try:
            d = self._read()
        except Exception:
            return self._json(400, {"error": "bad request"})
        try:
            if self.path.rstrip("/").endswith("/session"):
                return self._json(200, monitor.open_session(
                    d["agent_id"], d["mandate_id"],
                    bytes.fromhex(d["eph_pub"]), d["nonce"]))
            if self.path.rstrip("/").endswith("/nonce"):
                return self._json(200, monitor.issue_nonce(d["token"]))
            if self.path.rstrip("/").endswith("/authorize"):
                return self._json(200, monitor.authorize_transfer(
                    token=d["token"], nonce=d["nonce"], payee_wallet=d["payee_wallet"],
                    amount=float(d["amount"]), currency=d["currency"],
                    timestamp=d["timestamp"], signature_hex=d["signature"],
                    step_up_id=d.get("step_up_id", ""),
                    evidence=d.get("evidence") or {}))
        except monitor.MonitorError as e:
            # A refusal is an answer, not a server fault — and the reason is safe to speak.
            return self._json(200, {"state": "refused", "reason": str(e)})
        except KeyError as e:
            return self._json(400, {"error": f"missing {e}"})
        except Exception as e:
            return self._json(500, {"error": type(e).__name__})
        self._json(404, {"error": "no such endpoint"})

    def do_GET(self):
        if self.path.rstrip("/").endswith("/health"):
            inv = ledger.check_invariant()
            return self._json(200, {"ok": True, "book_balanced": inv["balanced"]})
        self._json(404, {"error": "no such endpoint"})

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    for mod in (ms, ledger, monitor, notify, trace, step_up):
        mod.init()
    print(f"monitor api at http://{a.host}:{a.port}", flush=True)
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
