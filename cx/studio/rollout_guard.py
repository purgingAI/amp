"""Rollout guard: switches the customer-app rollout OFF by itself when real answers go wrong.

Every minute it reads the last WINDOW of answers the Orchestrator gave real app customers (ledger,
source=app) and checks four things:
  unfaithful   the faithfulness check found sentences its documents do not support
  no-info      the agent had to say it has no information (the documents did not really cover it)
  too slow     longer than the engine waits (STUDIO_APP_TIMEOUT, 20 s) — the customer got the engine's
               answer and ours was wasted
  thumbs-down  a customer marked the answer unhelpful
With at least MIN_N answers in the window, any rate over its limit sets the rollout to 0% (an audited
change by "rollout-guard"), records why, and posts to Teams when STUDIO_TEAMS_WEBHOOK is set.
It only ever turns the rollout DOWN — raising it again is always a person's decision.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

import ledger
import store

FILE = "rollout_guard.json"
WINDOW = 3600
MIN_N = 10
SLOW_MS = int(float(os.environ.get("STUDIO_APP_TIMEOUT", "20")) * 1000)
LIMITS = {"unfaithful": 0.25, "noinfo": 0.35, "slow": 0.20, "down": 0.30}
LABELS = {"unfaithful": "went beyond their documents", "noinfo": "said they had no information",
          "slow": "were slower than the engine waits", "down": "got a thumbs-down"}


def metrics(window: int = WINDOW) -> dict:
    import faith
    since = time.time() - window
    rows = [r for r in ledger.recent(2000, "app") if r["at"] >= since]
    full = [ledger.get(r["id"]) for r in rows]
    n = len(full)
    scored = [r for r in full if r.get("faith_score") is not None]
    cnt = {
        "unfaithful": sum(1 for r in scored if r["faith_score"] < 0.67),
        "noinfo": sum(1 for r in full if faith.META.search(r.get("answer") or "")),
        "slow": sum(1 for r in full if (r.get("ms_total") or 0) > SLOW_MS),
        "down": sum(1 for r in full if r.get("feedback") == -1),
    }
    base = {"unfaithful": len(scored), "noinfo": n, "slow": n, "down": n}
    rate = {k: (cnt[k] / base[k]) if base[k] else 0.0 for k in cnt}
    ms = sorted(r.get("ms_total") or 0 for r in full)
    return {"n": n, "window_min": window // 60, "counts": cnt, "rates": {k: round(v, 3) for k, v in rate.items()},
            "scored": len(scored), "p50_ms": ms[len(ms) // 2] if ms else None, "p95_ms": ms[int(len(ms) * .95)] if ms else None}


def state() -> dict:
    return store.read(FILE) or {"events": [], "last": None}


def check() -> dict | None:
    """One pass. Returns the trip event if it switched the rollout off."""
    import learning
    st = state()
    pct = learning.settings()["rollout"]["percent"]
    m = metrics()
    st["last"] = {"at": time.time(), "percent": pct, **m}
    trip = None
    if pct > 0 and m["n"] >= MIN_N:
        over = [k for k, lim in LIMITS.items() if m["rates"][k] > lim and (k != "unfaithful" or m["scored"] >= MIN_N // 2)]
        if over:
            why = "; ".join(f"{round(100 * m['rates'][k])}% of answers {LABELS[k]} (limit {round(100 * LIMITS[k])}%)" for k in over)
            learning.save({"rollout": {"percent": 0}}, "rollout-guard")
            trip = {"at": time.time(), "from_percent": pct, "why": why, "metrics": m}
            st["events"] = (st.get("events") or [])[-19:] + [trip]
            _notify(trip)
            try:
                import bus
                bus.publish("guard", "rollout-guard", text=f"switched the customer-app rollout OFF (was {pct}%): {why}")
            except Exception:                               # noqa: BLE001
                pass
    store.write(FILE, st)
    return trip


def status() -> dict:
    st = state()
    return {"limits": LIMITS, "labels": LABELS, "min_n": MIN_N, "window_min": WINDOW // 60, "slow_ms": SLOW_MS,
            "last": st.get("last"), "events": (st.get("events") or [])[-5:]}


def _notify(trip: dict) -> None:
    url = os.environ.get("STUDIO_TEAMS_WEBHOOK")
    if not url:
        return
    try:
        body = json.dumps({"text": f"⚠ Agent Studio switched the customer-app rollout OFF (was {trip['from_percent']}%): {trip['why']}"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=10)
    except Exception:                                       # noqa: BLE001 — the switch-off itself already happened
        pass


def worker() -> None:
    def loop():
        while True:
            try:
                check()
            except Exception:                               # noqa: BLE001 — the guard must never die
                pass
            time.sleep(60)
    threading.Thread(target=loop, daemon=True, name="rollout-guard").start()
