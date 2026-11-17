"""A progress screen for the deployment, because "is it done yet" deserved a better answer.

The deploy's slow part is a 466MB model copy over a VPN, and the only visibility was me
polling by hand and guessing. This polls the target itself on a background thread and serves
what it finds, so the question answers itself.

Nothing here can change the deployment. It reads state and renders it.

    ~/amp/.venv/bin/python deploy_ui.py        then open http://<lan-ip>:8490
"""
from __future__ import annotations

import http.server
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

HOST = os.environ.get("CX_DEPLOY_HOST", "riley@127.0.0.1")
DIR = os.environ.get("CX_DEPLOY_DIR", "/home/riley/openwallet-cx")
PORT = int(os.environ.get("CX_DEPLOY_PORT", "8440"))
MODEL_MB = 466
POLL = 6.0

STATE: dict = {"at": 0, "steps": [], "model_mb": 0, "ports": {}, "probe": None,
               "log": "", "error": None}
_LOCK = threading.Lock()


def _ssh(cmd: str, timeout: int = 20) -> str:
    """One shot at the remote. Returns "" on any failure — this is a status page, and a
    failed check is itself a status, not an exception to propagate."""
    try:
        out = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             "-o", "StrictHostKeyChecking=accept-new", HOST, cmd],
            capture_output=True, text=True, timeout=timeout).stdout
        # The box prints post-quantum warnings on every connection; they are not output.
        return "\n".join(l for l in out.splitlines()
                         if not re.search(r"post-quantum|store now|openssh\.com/pq|"
                                          r"session may be vulnerable|may need to be upgraded",
                                          l)).strip()
    except Exception:
        return ""


def _probe() -> dict | None:
    """Ask the deployed assistant a real question. The only proof that matters."""
    ip = HOST.split("@")[-1]
    try:
        req = urllib.request.Request(
            f"http://{ip}:{PORT}/api/cx/ask", method="POST",
            data=json.dumps({"text": "how many bills are due",
                             "session": "deploy-ui"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read())
        return {"ok": True, "intent": d.get("intent"),
                "confidence": round((d.get("confidence") or 0) * 100),
                "say": (d.get("say") or "")[:160],
                "model": d.get("model") or ""}
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}"}


def poll() -> None:
    while True:
        try:
            # One round trip for everything cheap, so the page is not waiting on five.
            raw = _ssh(
                f"du -sm {DIR}/cx/model 2>/dev/null | cut -f1; echo ---; "
                f"for p in {PORT} 8450 8480; do printf '%s=' $p; "
                f"  ss -ltn 2>/dev/null | grep -q \":$p \" && echo up || echo down; done; echo ---; "
                f"[ -d {DIR}/cx ] && echo code=yes || echo code=no; "
                f"[ -x {DIR}/.venv/bin/python ] && echo venv=yes || echo venv=no; "
                f"[ -x {DIR}/.venv-model/bin/python ] && echo model_venv=yes || echo model_venv=no; "
                f"[ -s {DIR}/cx/model/openwallet-cx-intent-ml-v5/model.safetensors ] "
                f"  && echo weights=yes || echo weights=no; echo ---; "
                f"tail -4 /tmp/cx-assist.log 2>/dev/null")
            if not raw:
                with _LOCK:
                    STATE["error"] = "cannot reach the server (VPN down?)"
                    STATE["at"] = time.time()
                time.sleep(POLL)
                continue

            parts = raw.split("---")
            mb = 0
            try:
                mb = int((parts[0] or "0").strip().splitlines()[0])
            except Exception:
                pass
            ports = dict(l.split("=") for l in (parts[1] or "").strip().splitlines()
                         if "=" in l)
            flags = dict(l.split("=") for l in (parts[2] or "").strip().splitlines()
                         if "=" in l)
            log = (parts[3] or "").strip() if len(parts) > 3 else ""

            steps = [
                ("Code uploaded", flags.get("code") == "yes"),
                ("App environment", flags.get("venv") == "yes"),
                ("Model environment (torch)", flags.get("model_venv") == "yes"),
                ("Model weights copied", flags.get("weights") == "yes"),
                ("Bill lookup stand-in", ports.get("8480") == "up"),
                ("Model sidecar", ports.get("8450") == "up"),
                ("Assistant", ports.get(str(PORT)) == "up"),
            ]
            probe = _probe() if ports.get(str(PORT)) == "up" else None
            with _LOCK:
                STATE.update(at=time.time(), steps=steps, model_mb=mb, ports=ports,
                             probe=probe, log=log, error=None)
        except Exception as exc:
            with _LOCK:
                STATE["error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(POLL)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deploying</title>
<style>
  :root{color-scheme:light dark;
    --bg:#f6f7f9;--ink:#1f2430;--muted:#6b7280;--line:#e6e9ee;--panel:#fff;
    --ok:#1b7f3b;--okbg:#e7f7ed;--wait:#9a5b00;--waitbg:#fff4e2;--bad:#b42318;--badbg:#fdeaea;
    --ui:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  @media (prefers-color-scheme:dark){:root{
    --bg:#0e1116;--ink:#e8eaee;--muted:#9aa3af;--line:#272c35;--panel:#161a21;
    --ok:#5fcf85;--okbg:#14301f;--wait:#f0b84a;--waitbg:#33280f;--bad:#ff8a8a;--badbg:#331a1c}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--ui);
    display:flex;justify-content:center;padding:18px 14px 40px}
  .wrap{width:min(560px,100%)}
  h1{font-size:19px;margin:0 0 2px;font-weight:750}
  .sub{color:var(--muted);font-size:12.5px;margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:15px 16px;margin-bottom:12px}
  .bar{height:9px;border-radius:99px;background:var(--line);overflow:hidden;margin:10px 0 7px}
  .bar i{display:block;height:100%;background:linear-gradient(90deg,#ef1a23,#f18918,#fbc007);
    transition:width .6s cubic-bezier(.3,.8,.3,1)}
  .pct{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
  .step{display:flex;align-items:center;gap:10px;padding:8px 0;border-top:1px solid var(--line)}
  .step:first-child{border-top:0}
  .dot{width:19px;height:19px;border-radius:50%;flex:0 0 auto;display:grid;place-items:center;
    font-size:11px;font-weight:800}
  .dot.y{background:var(--okbg);color:var(--ok)}
  .dot.n{background:var(--waitbg);color:var(--wait)}
  .step b{font-weight:550;font-size:14px}
  .step span{margin-left:auto;font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
  .tag{display:inline-block;font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;
    padding:2px 8px;border-radius:99px;font-weight:700}
  .tag.y{background:var(--okbg);color:var(--ok)} .tag.n{background:var(--badbg);color:var(--bad)}
  .tag.w{background:var(--waitbg);color:var(--wait)}
  pre{margin:8px 0 0;font:11.5px/1.5 ui-monospace,Menlo,monospace;color:var(--muted);
    white-space:pre-wrap;word-break:break-word}
  a{color:#e5472c;font-weight:600;text-decoration:none}
  .hint{font-size:11.5px;color:var(--muted);margin-top:14px;text-align:center}
</style></head><body><div class="wrap">
  <h1>OpenWallet CX assistant</h1>
  <div class="sub" id="where">deploying to &hellip;</div>
  <div class="card">
    <div style="display:flex;align-items:baseline;gap:8px">
      <b style="font-size:26px;font-weight:800" id="pct">0%</b>
      <span style="color:var(--muted);font-size:12.5px" id="phase">starting</span>
      <span class="tag w" id="live" style="margin-left:auto">checking</span>
    </div>
    <div class="bar"><i id="fill" style="width:0"></i></div>
    <div class="pct"><span id="mb"></span><span id="age"></span></div>
  </div>
  <div class="card" id="steps"></div>
  <div class="card" id="probe" style="display:none"></div>
  <div class="card" id="logcard" style="display:none"><b style="font-size:12.5px">Server log</b><pre id="log"></pre></div>
  <div class="hint">refreshes every few seconds &middot; this page only reads, it changes nothing</div>
</div>
<script>
const $ = i => document.getElementById(i);
async function tick(){
  let d; try{ d = await fetch('state').then(r=>r.json()); }catch(e){ return; }
  $('where').textContent = 'deploying to ' + d.host + ' \\u00b7 ' + d.url;

  const done = d.steps.filter(s=>s[1]).length, total = d.steps.length;
  // The model copy is most of the wall-clock time, so it gets weight in the number rather
  // than counting as one step of seven. A progress bar that sits at 43% for an hour is
  // worse than no progress bar.
  const copy = Math.min(1, d.model_mb / d.model_total);
  const weighted = Math.round(((done / total) * 0.55 + copy * 0.45) * 100);
  $('pct').textContent = (d.ready ? 100 : Math.min(99, weighted)) + '%';
  $('fill').style.width = (d.ready ? 100 : Math.min(99, weighted)) + '%';
  $('phase').textContent = d.ready ? 'ready' : (d.phase || '');
  $('mb').textContent = d.model_mb ? ('model ' + d.model_mb + ' / ' + d.model_total + ' MB') : '';
  $('age').textContent = d.error ? d.error : ('checked ' + d.age + 's ago');
  const live = $('live');
  live.className = 'tag ' + (d.error ? 'n' : d.ready ? 'y' : 'w');
  live.textContent = d.error ? 'unreachable' : d.ready ? 'live' : 'working';

  $('steps').innerHTML = d.steps.map(([name, ok]) =>
    `<div class="step"><i class="dot ${ok?'y':'n'}">${ok?'\\u2713':'\\u00b7'}</i>`
    + `<b>${name}</b><span>${ok?'done':'pending'}</span></div>`).join('');

  if (d.probe){
    const p = $('probe'); p.style.display='';
    p.innerHTML = d.probe.ok
      ? `<b style="font-size:12.5px">Answering</b>`
        + `<pre>&gt; how many bills are due\n${d.probe.say}`
        + `\n\n[${d.probe.intent} ${d.probe.confidence}% \u00b7 ${d.probe.model||'rules'}]</pre>`
      : `<b style="font-size:12.5px">Not answering yet</b><pre>${d.probe.why}</pre>`;
  }
  if (d.log){ $('logcard').style.display=''; $('log').textContent = d.log; }
  if (d.ready) $('where').innerHTML = 'ready \\u00b7 <a href="'+d.url+'">open the assistant</a>';
}
tick(); setInterval(tick, 4000);
</script></body></html>
"""


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/state":
            with _LOCK:
                s = dict(STATE)
            ip = HOST.split("@")[-1]
            ready = bool(s["ports"].get(str(PORT)) == "up"
                         and (s.get("probe") or {}).get("ok"))
            pend = [n for n, ok in s["steps"] if not ok]
            return self._send(200, json.dumps({
                "host": ip, "url": f"http://{ip}:{PORT}/assist",
                "steps": s["steps"], "model_mb": s["model_mb"], "model_total": MODEL_MB,
                "probe": s["probe"], "log": s["log"], "error": s["error"],
                "ready": ready, "phase": (pend[0] if pend else "finishing"),
                "age": int(time.time() - s["at"]) if s["at"] else 0}))
        self._send(404, json.dumps({"error": "not found"}))


def main() -> None:
    threading.Thread(target=poll, daemon=True).start()
    port = int(os.environ.get("CX_DEPLOY_UI_PORT", "8490"))
    print(f"  deployment progress on http://0.0.0.0:{port}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
