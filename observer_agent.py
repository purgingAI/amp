#!/usr/bin/env python3
"""Laptop 3 — the observer. Streams the live trace so the flow can be watched as it happens.

Run:   ./observer_agent.py --port 8200

Read-only by construction: it serves a page and a trace stream and has no route that can move,
approve or alter anything. A screen shown to a room full of people should not be able to spend.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import trace, ledger, mandate_store as ms, monitor, notify   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "observer.html")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        # Match on the path alone. "/?demo=1" is the same page as "/", and comparing
        # the raw request target meant any query string 404'd.
        path = urlparse(self.path).path
        if path.startswith("/events"):
            return self._events()
        if path in ("/", "/index.html"):
            return self._page()
        self.send_error(404)

    def _page(self):
        try:
            body = open(PAGE, "rb").read()
        except OSError:
            self.send_error(404, "observer.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def _wallets(self):
        out = {}
        with ms.conn() as c:
            rows = c.execute("SELECT DISTINCT account_id, owner FROM accounts").fetchall()
        for r in rows:
            if r["account_id"] != "acc_funding":
                out[r["account_id"]] = {"owner": r["owner"],
                                        "balance": ledger.balance(r["account_id"])}
        return out

    def _events(self):
        self.send_response(200)
        for h, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                     ("Connection", "keep-alive"), ("Access-Control-Allow-Origin", "*")):
            self.send_header(h, v)
        self.end_headers()
        try:
            self._send({"type": "hello", "stages": trace.STAGES,
                        "wallets": self._wallets()})
            # Start at the end: a demo should open on a quiet graph, not replay history.
            seq = max([t["trace_seq"] for t in trace.since(0, 5000)] or [0])
            while True:
                rows = trace.since(seq, 200)
                for t in rows:
                    seq = t["trace_seq"]
                    if t["nonce"].startswith("session:"):
                        continue
                    self._send({"type": "step", **t})
                if rows:
                    self._send({"type": "wallets", "wallets": self._wallets()})
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    for mod in (ms, ledger, monitor, notify, trace):
        mod.init()
    print(f"observer at http://{a.host}:{a.port}", flush=True)
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
