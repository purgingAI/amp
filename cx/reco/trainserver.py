"""Live training server: run a training job and stream every step to the browser.

Training is usually a wall of scrolling numbers that you either trust or do not. This serves
the same run as events — the loss as it moves, the penalty applied to each individual
example, what the model decided each feature was worth, and the comparison against a
baseline — so "the model is learning" is something you can watch and disagree with rather
than something you are told.

    ~/amp/laya/.venv/bin/python trainserver.py --port 8460      # then open /
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import http.server
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

SUBS: list[queue.Queue] = []
SUBS_LOCK = threading.Lock()
HISTORY: list[dict] = []
RUNNING = threading.Event()


def emit(kind: str, payload: dict) -> None:
    ev = {"t": time.time(), "ts": time.strftime("%H:%M:%S"), "kind": kind, **payload}
    HISTORY.append(ev)
    with SUBS_LOCK:
        for q in list(SUBS):
            q.put(ev)


def start_run(which: str) -> None:
    if RUNNING.is_set():
        return
    RUNNING.set()
    HISTORY.clear()

    def go():
        try:
            if which == "rules":
                import train_rules
                train_rules.run(emit)
            else:
                import train
                train.run(emit, with_minilm=(which != "fast"))
        except Exception as e:
            emit("run.failed", {"error": f"{type(e).__name__}: {e}"})
        finally:
            RUNNING.clear()

    threading.Thread(target=go, daemon=True).start()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/train"):
            try:
                body = open(os.path.join(WEB, "train.html"), "rb").read()
            except OSError:
                return self._send(404, "page missing", "text/plain")
            return self._send(200, body, "text/html; charset=utf-8")
        if path == "/events":
            return self.events()
        if path == "/api/history":
            return self._send(200, json.dumps({"events": HISTORY,
                                               "running": RUNNING.is_set()}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/run":
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if RUNNING.is_set():
                return self._send(200, json.dumps({"ok": False, "error": "already running"}))
            start_run(req.get("which") or "fast")
            return self._send(200, json.dumps({"ok": True}))
        self._send(404, json.dumps({"error": "not found"}))

    def events(self):
        q = queue.Queue()
        with SUBS_LOCK:
            SUBS.append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for ev in list(HISTORY):
                self.wfile.write(f"data: {json.dumps(dict(ev, replay=True))}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=20)
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with SUBS_LOCK:
                if q in SUBS:
                    SUBS.remove(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8460)
    args = ap.parse_args()
    print(f"  training theatre on http://{args.host}:{args.port}", flush=True)
    http.server.ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
