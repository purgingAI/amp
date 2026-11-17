#!/usr/bin/env python3
"""Agent B — the payee's laptop. Serves the wallet screen and streams credited events.

Run on laptop 2:   ./payee_agent.py --agent agt_alex --port 8100

Reads the outbox rather than being called directly, so a transfer that captured while this
laptop was asleep still shows up the moment it starts. Balances always come from the ledger,
never from the event stream — an event describes money that already moved; it is not the
record of it.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import notify, mandate_store as ms, ledger   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "payee.html")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    agent_id = "agt_alex"

    def do_GET(self):
        if self.path.startswith("/events"):
            return self._events()
        if self.path in ("/", "/index.html"):
            return self._page()
        self.send_error(404)

    def _page(self):
        try:
            body = open(PAGE, "rb").read()
        except OSError:
            self.send_error(404, "payee.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def _events(self):
        self.send_response(200)
        for h, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                     ("Connection", "keep-alive"), ("Access-Control-Allow-Origin", "*")):
            self.send_header(h, v)
        self.end_headers()

        who = notify.balance_of(self.agent_id)
        name = ""
        for m in _mandates_for(self.agent_id):
            name = m.payee.display_name or self.agent_id
            break
        try:
            self._send({"type": "hello", "agent_id": self.agent_id, "name": name,
                        "wallet": who.get("wallet", ""), "currency": who.get("currency", ""),
                        "balance": who.get("balance", 0)})
            # Replay what arrived while this laptop was away, then follow live.
            since = 0
            while True:
                for ev in notify.poll(self.agent_id, since):
                    since = ev["event_id"]
                    if ev["kind"] != "credited":
                        continue
                    bal = ledger.balance(ev["payload"]["to_wallet"])
                    self._send({"type": "credited", "payload": ev["payload"], "balance": bal})
                    notify.ack(ev["event_id"])
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


def _mandates_for(agent_id):
    from app.mandate import Mandate
    with ms.conn() as c:
        rows = c.execute("SELECT doc FROM mandates WHERE payee_agent=? ORDER BY updated DESC",
                         (agent_id,)).fetchall()
    return [Mandate.from_dict(json.loads(r["doc"])) for r in rows]


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="agt_alex")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    ms.init(); ledger.init(); notify.init()
    Handler.agent_id = a.agent
    srv = Server((a.host, a.port), Handler)
    print(f"wallet for {a.agent} at http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
