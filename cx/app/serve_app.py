#!/usr/bin/env python3
"""The OpenWallet customer-app demo, over HTTPS, so phones on the LAN can use the microphone.

Serves this folder and forwards a fixed allow-list of API calls to the openwallet server
(127.0.0.1:8440), which owns every decision:

    GET  /api/cx/home   balance, recipients, recent payments (read-only)
    POST /api/cx/ask    one customer turn -> the engine's answer
    POST /api/cx/act    a tap on a card (confirm / cancel / PIN) -> the PIN-gated path
    POST /api/stt       speech -> text (whisper, English / Arabic / Hindi)
    GET  /events        live payment stages, streamed (for the "sent" result)
    GET  /ca.crt        this demo's local CA, to install on a phone (no browser warning)

Nothing else is reachable through this server. Money still only moves through the openwallet
server's own PIN gate — this process cannot authorise anything.

    python3 serve_app.py [--port 8443] [--engine http://127.0.0.1:8440]
"""
from __future__ import annotations

import argparse
import os
import ssl
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = "http://127.0.0.1:8440"
GET_OK = {"/api/cx/home", "/events"}
POST_OK = {"/api/cx/ask", "/api/cx/act", "/api/stt", "/api/cx/login", "/api/cx/session",
           "/api/auth/otp", "/api/auth/verify", "/api/auth/pin", "/api/auth/logout"}
MAX_BODY = 12 * 1024 * 1024          # a minute of 16 kHz WAV is ~2 MB; this is plenty


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        # the app must never be served stale during a demo
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _deny(self):
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"not found"}')

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ca.crt":
            data = open(os.path.join(HERE, "certs", "ca.crt"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-x509-ca-cert")
            self.send_header("Content-Disposition", 'attachment; filename="openwallet-demo-ca.crt"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/certs/"):
            return self._deny()                   # the keys are never served
        if path.startswith("/api/") or path == "/events":
            if path not in GET_OK:
                return self._deny()
            return self._forward("GET", stream=(path == "/events"))
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] not in POST_OK:
            return self._deny()
        return self._forward("POST")

    def _forward(self, method: str, stream: bool = False):
        body = None
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self.send_response(413)
                self.end_headers()
                return
            body = self.rfile.read(n)
        headers = {"Content-Type": self.headers.get("Content-Type") or "application/json"}
        # The logged-in customer's app token. The engine maps it to their wallet login; without
        # it, every customer-data call is refused.
        if self.headers.get("X-Cx-Auth"):
            headers["X-Cx-Auth"] = self.headers["X-Cx-Auth"][:200]
        req = urllib.request.Request(ENGINE + self.path, data=body, headers=headers, method=method)
        try:
            up = urllib.request.urlopen(req, timeout=None if stream else 180)
        except urllib.error.HTTPError as e:
            up = e
        except (urllib.error.URLError, OSError) as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(('{"ok":false,"error":"the OpenWallet service is not running (%s)"}'
                              % type(e).__name__).encode())
            return
        self.send_response(getattr(up, "status", 200))
        self.send_header("Content-Type", up.headers.get("Content-Type") or "application/json")
        if stream:
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    line = up.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        data = up.read()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--engine", default=ENGINE)
    a = ap.parse_args()
    ENGINE = a.engine.rstrip("/")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(os.path.join(HERE, "certs", "server.crt"), os.path.join(HERE, "certs", "server.key"))
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"OpenWallet app on https://{a.host}:{a.port}/  ·  engine {ENGINE}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
