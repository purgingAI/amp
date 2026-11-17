"""The Action agent's brain: the intent model versions (v9 … v24), their measured accuracy, and
switching which one is live — the same steps done by hand until now:

  1. change LIVE_MODEL in the launcher (laptop: ~/amp/up.sh · server: ~/amp/deploy/run_server.sh)
  2. restart ONLY the model service (port 8450); the engine keeps running and uses its keyword
     fallback for the ~30 s the new model takes to load
  3. wait for /health to report the new version — if it does not within the time limit, put the
     old version back and restart it (automatic rollback)

Accuracy comes from the model journal (learn/models/<run>.json — written after every evaluation)
plus the full-engine blind-set results, which live in BLIND below because each round was measured
on a NEW blind set (the honest number for that round).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

import store

CX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(CX, "model")
JOURNAL = os.path.join(CX, "docs", "model-brief", "learn", "models")
LAPTOP = os.path.expanduser("~/amp/up.sh")
SERVER = os.path.expanduser("~/amp/deploy/run_server.sh")
INTENT = "http://127.0.0.1:8450/health"
PREFIX = "openwallet-cx-intent-ml-"

#: Full engine (rules + model) on fresh blind sets nobody trained on — from each round's evaluation.
BLIND = {
    "v16": {"set": "blind_v2", "n": 612, "score": 83.5, "leaks": 6},
    "v21": {"set": "blind_v2", "n": 612, "score": 88.2, "leaks": 3},
    "v22": {"set": "blind_v3", "n": None, "score": 93.0, "leaks": 6, "note": "inflated — rules were tuned on its misses"},
    "v23": {"set": "blind_v5", "n": 700, "score": 92.1, "leaks": 19, "also": {"set": "blind_v4", "n": 640, "score": 90.2, "leaks": 9}},
    "v24": {"set": "blind_v5", "n": 700, "score": 93.7, "leaks": 14},
}

_lock = threading.Lock()
STATUS = "action_switch.json"


def launcher() -> str | None:
    return LAPTOP if os.path.exists(LAPTOP) else SERVER if os.path.exists(SERVER) else None


def live_run() -> str | None:
    f = launcher()
    if not f:
        return None
    m = re.search(r'LIVE_MODEL="?\$\{CX_LIVE_MODEL:-' + PREFIX + r'(v\d+)\}', open(f).read())
    return m.group(1) if m else None


def _num(run: str) -> int:
    return int(re.sub(r"\D", "", run) or 0)


def models() -> dict:
    live = live_run()
    runs = {}
    for p in glob.glob(os.path.join(JOURNAL, "v*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run = d.get("run") or os.path.basename(p)[:-5]
        h, b, s = d.get("heldout") or {}, d.get("brd") or {}, d.get("story") or {}
        runs[run] = {
            "run": run, "title": d.get("title") or s.get("title"), "date": d.get("date"),
            "changed": s.get("changed"), "lesson": s.get("lesson"), "wrong": s.get("wrong"),
            "heldout": round(100 * h["ok"] / h["n"], 1) if h.get("n") else None, "heldout_n": h.get("n"),
            "leaks": h.get("leaks"), "refused": h.get("refused"),
            "by_lang": {k: round(100 * v["ok"] / v["n"], 1) for k, v in (h.get("by_lang") or {}).items() if v.get("n")},
            "brd": round(100 * b["ok"] / b["n"], 1) if b.get("n") else None,
            "temperature": (d.get("calibration") or {}).get("temperature"),
            "params_m": (d.get("card") or {}).get("params_millions"),
            "train_rows": ((d.get("card") or {}).get("trained_on") or {}).get("train"),
            "gate": (d.get("decision") or {}).get("summary"),
        }
    for p in glob.glob(os.path.join(MODELS, PREFIX + "v*")):
        run = os.path.basename(p)[len(PREFIX):]
        r = runs.setdefault(run, {"run": run, "title": None})
        r["on_disk"] = os.path.isfile(os.path.join(p, "onnx", "model.onnx")) or os.path.isfile(os.path.join(p, "model.safetensors"))
        try:
            r["temperature"] = r.get("temperature") or json.load(open(os.path.join(p, "calibration.json"))).get("temperature")
        except (OSError, ValueError):
            pass
    for run, r in runs.items():
        r["journal"] = r.get("heldout") is not None or r.get("title") is not None
        # only journaled versions (v9+) share today's label set; earlier ones could break the engine
        r["switchable"] = bool(r.get("on_disk")) and r["journal"] and _num(run) >= 9
        r["blind"] = BLIND.get(run)
        try:
            jd = json.load(open(os.path.join(JOURNAL, f"{run}.json"), encoding="utf-8"))
        except (OSError, ValueError):
            jd = {}
        if not r["blind"] and (jd.get("blind_model") or {}).get("score") is not None:
            r["blind"] = {"set": jd["blind_model"]["set"], "score": jd["blind_model"]["score"], "n": jd["blind_model"].get("n")}
        st = jd.get("studio") or {}
        if st.get("challenger"):
            r["challenger"] = {"gates": st.get("gates", []), "passed": all(g["pass"] for g in st.get("gates", [])), "labels": st.get("human_labels"),
                               "exams": {k: {s: v[s]["acc"] for s in v} for k, v in (st.get("exams") or {}).items()}}
        r["live"] = run == live
        r.setdefault("on_disk", False)
    out = sorted(runs.values(), key=lambda r: _num(r["run"]))
    return {"live": live, "machine": "laptop" if launcher() == LAPTOP else "server", "models": out,
            "switch": store.read(STATUS), "history": store.read("action_history.json", [])}


def _health() -> dict | None:
    try:
        return json.loads(urllib.request.urlopen(INTENT, timeout=4).read())
    except Exception:                                       # noqa: BLE001
        return None


def _set_launcher(run: str) -> str:
    f = launcher()
    s = open(f).read()
    new, n = re.subn(r'(LIVE_MODEL="?\$\{CX_LIVE_MODEL:-)' + PREFIX + r'v\d+(\})', r"\g<1>" + PREFIX + run + r"\g<2>", s, count=1)
    if not n:
        raise RuntimeError("LIVE_MODEL line not found in the launcher")
    tmp = f + ".studio.tmp"
    open(tmp, "w").write(new)
    os.chmod(tmp, os.stat(f).st_mode)
    os.replace(tmp, f)
    return f


def _restart_intent() -> None:
    f = launcher()
    if f == LAPTOP:
        subprocess.run(["systemctl", "--user", "stop", "openwallet-intent"], timeout=60)
        subprocess.run([LAPTOP], timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if re.search(r":8450\s", line):
            for pid in re.findall(r"pid=(\d+)", line):
                try:
                    os.kill(int(pid), 15)
                except OSError:
                    pass
    time.sleep(3)
    subprocess.run([SERVER], timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _set_status(**kw) -> None:
    s = store.read(STATUS, {}) or {}
    s.update(kw, updated=time.time())
    store.write(STATUS, s)


def switch(run: str, by: str) -> dict:
    """Start switching the live intent model (in the background); poll status() for progress."""
    with _lock:
        cur = store.read(STATUS) or {}
        if cur.get("state") == "running" and time.time() - cur.get("updated", 0) < 300:
            raise RuntimeError("a switch is already in progress")
        m = next((x for x in models()["models"] if x["run"] == run), None)
        if not m or not m.get("on_disk"):
            raise ValueError(f"{run} is not on this machine")
        if not m.get("switchable"):
            raise ValueError(f"{run} predates the model journal and today's label set — it cannot be made live safely")
        old = live_run()
        if old == run:
            raise ValueError(f"{run} is already live")
        store.write(STATUS, {"state": "running", "from": old, "to": run, "by": by, "started": time.time(), "steps": [], "updated": time.time()})
    threading.Thread(target=_do_switch, args=(old, run, by), daemon=True).start()
    return {"ok": True, "from": old, "to": run}


def _step(text: str, ok: bool | None = None) -> None:
    s = store.read(STATUS, {}) or {}
    s.setdefault("steps", []).append({"t": time.time(), "text": text, "ok": ok})
    s["updated"] = time.time()
    store.write(STATUS, s)


def _wait_for(run: str, limit: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit:
        h = _health()
        if h and h.get("ready") and ((h.get("card") or {}).get("name") or "").endswith(run):
            return True
        time.sleep(3)
    return False


def _do_switch(old: str, run: str, by: str) -> None:
    try:
        f = _set_launcher(run)
        _step(f"launcher updated ({os.path.basename(f)}): live model → {run}", True)
        _step("restarting the model service — the engine answers with its keyword fallback meanwhile")
        _restart_intent()
        _step(f"loading {run} (about 30 seconds)…")
        if _wait_for(run, 180):
            h = _health() or {}
            _step(f"{run} is answering — health check passed", True)
            hist = store.read("action_history.json", []) or []
            hist.append({"from": old, "to": run, "by": by, "at": time.time()})
            store.write("action_history.json", hist[-50:])
            _set_status(state="done", finished=time.time(), card=(h.get("card") or {}).get("name"))
            return
        _step(f"{run} did not come up in time — rolling back to {old}", False)
        _set_launcher(old)
        _restart_intent()
        back = _wait_for(old, 180)
        _step(f"{old} is answering again" if back else f"{old} is not answering either — check the server", back)
        _set_status(state="rolled_back" if back else "failed", finished=time.time())
    except Exception as exc:                                # noqa: BLE001
        _step(f"error: {type(exc).__name__}: {exc}", False)
        try:
            if old and live_run() != old:
                _set_launcher(old)
                _restart_intent()
        except Exception:                                   # noqa: BLE001
            pass
        _set_status(state="failed", finished=time.time(), error=str(exc))
