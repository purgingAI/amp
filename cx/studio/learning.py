"""Learning switches and the nightly scheduler — so the system improves only when you want it to.

Two systems learn, each with its own switches:
  orchestrator   the reflex that picks an agent           (trains in ~1 minute)
  intent         the action agent's intent model (v24…)   (trains in ~15–40 minutes)

  shadow    (one switch for both) mirror real app traffic into the ledger and the Teach-me inbox
  nightly   retrain at `time` (UAE) — ONLY when all of these hold:
              · the live version's score is below `target` (e.g. 95%) — at or above it, nothing runs
              · at least `min_new_labels` new HUMAN labels since the last run
              · no other training job is running, and there is disk space
  Nothing is ever promoted by the scheduler: it trains a challenger, runs the gates, and waits for a
  person to press Promote. Production can run with nightly OFF and capture ON (or both off).
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import threading
import time

import store

try:
    from zoneinfo import ZoneInfo
    UAE = ZoneInfo("Asia/Dubai")
except Exception:                                           # noqa: BLE001
    UAE = dt.timezone(dt.timedelta(hours=4))

DEFAULTS = {
    "shadow": {"enabled": True},
    # the Orchestrator answering REAL app customers — knowledge agents only, engine is the fallback
    "rollout": {"percent": 0},
    "orchestrator": {"nightly": False, "time": "02:30", "target": 95.0, "min_new_labels": 20},
    "intent": {"nightly": False, "time": "03:00", "target": 95.0, "min_new_labels": 50},
    # the Autopilot: observe → diagnose → stage examples → train → court → red-team → briefing. Never promotes.
    "autopilot": {"enabled": False, "time": "03:30", "max_examples": 24},
}
FILE = "learning.json"
_lock = threading.Lock()


def settings() -> dict:
    s = store.read(FILE) or {}
    out = {}
    for k, d in DEFAULTS.items():
        out[k] = {**d, **(s.get(k) or {})}
    out["runs"] = s.get("runs", {})
    return out


def save(new: dict, by: str) -> dict:
    with _lock:
        cur = settings()
        for k, d in DEFAULTS.items():
            for f, v in (new.get(k) or {}).items():
                if f not in d:
                    continue
                if isinstance(d[f], bool):
                    cur[k][f] = bool(v)
                elif isinstance(d[f], float):
                    cur[k][f] = max(50.0, min(99.9, float(v)))
                elif f == "percent":
                    cur[k][f] = max(0, min(100, int(v)))
                elif isinstance(d[f], int):
                    cur[k][f] = max(1, min(10000, int(v)))
                elif f == "time" and isinstance(v, str) and len(v) == 5 and v[2] == ":":
                    cur[k][f] = v
        cur.setdefault("audit", [])
        cur["audit"] = (store.read(FILE) or {}).get("audit", [])[-49:] + [{"at": time.time(), "by": by, "change": new}]
        store.write(FILE, cur)
        return settings()


def _note_run(system: str, **kw) -> None:
    s = store.read(FILE) or {}
    s.setdefault("runs", {})[system] = {"at": time.time(), **kw}
    store.write(FILE, s)


def live_score(system: str) -> float | None:
    """The number the target is compared with: the live version's score on unseen sentences."""
    if system == "orchestrator":
        v = store.versions()["router"]
        x = next((x for x in v["list"] if x["id"] == v["active"]), None)
        return (x or {}).get("scores", {}).get("score")
    import actions
    m = next((m for m in actions.models()["models"] if m["live"]), None) or {}
    b = m.get("blind") or {}
    return b.get("score") if b.get("score") is not None else m.get("heldout")


def due(system: str) -> dict:
    """Would tonight's run start — and if not, why not."""
    import inbox
    import jobs
    s = settings()[system]
    score, new = live_score(system), inbox.new_labels_since(system, settings()["runs"].get(system, {}).get("at", 0))
    reasons = []
    if not s["nightly"]:
        reasons.append("nightly training is switched off")
    if score is not None and score >= s["target"]:
        reasons.append(f"the live score {score:g} is already at or above the target {s['target']:g}")
    if new < s["min_new_labels"]:
        reasons.append(f"only {new} new human label(s) — needs {s['min_new_labels']}")
    if jobs.running():
        reasons.append("another training job is running")
    free = shutil.disk_usage(store.DATA).free / 2**30
    need = 3.0 if system == "intent" else 0.5
    if free < need:
        reasons.append(f"only {free:.1f} GB disk free — needs {need:g} GB")
    return {"system": system, "would_run": not reasons, "reasons": reasons, "live_score": score, "target": s["target"], "new_labels": new}


def next_run(system: str) -> float | None:
    s = settings()[system]
    if not s["nightly"]:
        return None
    hh, mm = (int(x) for x in s["time"].split(":"))
    now = dt.datetime.now(UAE)
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def status() -> dict:
    s = settings()
    return {"settings": {k: s[k] for k in DEFAULTS}, "runs": s["runs"],
            "systems": {k: dict(due(k), next_run=next_run(k)) for k in ("orchestrator", "intent")}}


def start(system: str, by: str, reason: str) -> dict:
    """Start a challenger training job (used by the scheduler and the 'Train challenger now' button)."""
    import jobs
    if system not in ("orchestrator", "intent"):
        raise ValueError("system must be orchestrator or intent")
    cfg = {"kind": "router_tune" if system == "orchestrator" else "intent_tune", "by": by, "challenger": True,
           "auto_promote": False, "threads": 4, "reason": reason}
    r = jobs.start(cfg)
    _note_run(system, job=r["id"], by=by, reason=reason)
    return r


def _tick() -> None:
    s = settings()
    now = dt.datetime.now(UAE)
    for system in ("orchestrator", "intent"):
        cfg = s[system]
        if not cfg["nightly"]:
            continue
        hh, mm = (int(x) for x in cfg["time"].split(":"))
        if (now.hour, now.minute) != (hh, mm):
            continue
        last = s["runs"].get(system, {}).get("at", 0)
        if time.time() - last < 3600:                       # already handled this minute/hour
            continue
        d = due(system)
        if d["would_run"]:
            try:
                start(system, "nightly scheduler", f"{d['new_labels']} new labels; live {d['live_score']} < target {d['target']}")
            except Exception as exc:                        # noqa: BLE001
                _note_run(system, skipped=f"could not start: {exc}")
        else:
            _note_run(system, skipped="; ".join(d["reasons"]))


def scheduler() -> None:
    def loop():
        while True:
            try:
                _tick()
            except Exception:                               # noqa: BLE001 — the scheduler must never die
                pass
            time.sleep(30)
    threading.Thread(target=loop, daemon=True, name="nightly").start()
