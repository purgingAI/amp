"""Calibrated confidence: make "60% sure" mean "right 60% of the time".

Fitted on sentences people have labelled — the golden sets plus the Teach-me labels:
  reflex      temperature scaling: p ∝ exp(log p / T), T chosen to minimise the log-loss of the right agent
  reasoning   per checkpoint (english / multilingual), Platt scaling of its confidence against how often
              its pick was actually right: p' = σ(a·logit(p) + b)
Each comes with a reliability table (confidence bins vs real accuracy) and ECE before/after.
Nothing changes routing until a person switches "use calibrated confidence" on; with too few labels
(< MIN per brain) that brain is left as it is.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

import store

FILE = "calibration.json"
MIN = 15
BINS = 5
_lock = threading.Lock()


def _bins(conf, ok) -> tuple[list[dict], float]:
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    out, ece, n = [], 0.0, max(1, len(conf))
    for b in range(BINS):
        lo, hi = b / BINS, (b + 1) / BINS
        m = (conf >= lo) & ((conf < hi) | (b == BINS - 1 and conf <= hi))
        if m.sum():
            acc, c = float(ok[m].mean()), float(conf[m].mean())
            ece += m.sum() / n * abs(acc - c)
            out.append({"lo": lo, "hi": hi, "n": int(m.sum()), "conf": round(c, 3), "acc": round(acc, 3)})
        else:
            out.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
    return out, round(float(ece), 4)


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _platt(conf, ok):
    """Fit a, b in σ(a·logit(c)+b) by Newton steps on the log-loss (two parameters, tiny data)."""
    x = np.array([_logit(c) for c in conf]); y = np.asarray(ok, float)
    a, b = 1.0, 0.0
    for _ in range(60):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([((p - y) * x).sum(), (p - y).sum()]) + 0.01 * np.array([a - 1, b])   # slight pull to identity
        w = p * (1 - p)
        H = np.array([[(w * x * x).sum() + 0.01, (w * x).sum()], [(w * x).sum(), w.sum() + 0.01]])
        try:
            a, b = np.array([a, b]) - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return float(a), float(b)


def labelled() -> list[tuple[str, str]]:
    import gates
    import inbox
    rows = [(g["text"], g["agent"]) for g in gates.golden()]
    rows += [(l["text"], l["agent"]) for l in inbox.training_rows("orchestrator")]
    seen, out = set(), []
    for t, a in rows:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append((t, a))
    return out


def fit(by: str) -> dict:
    import router
    g = router.graph()
    cands = router.candidates(g)
    rows = [(t, a) for t, a in labelled() if any(c["id"] == a for c in cands)]
    store.write(FILE, dict(get(), state="fitting", started=time.time(), n=len(rows)))
    res = {"at": time.time(), "by": by, "n": len(rows), "applied": get().get("applied", False), "state": "done"}
    # reflex: temperature on the live head
    h = router.head()
    if h:
        P, Y = [], []
        for t, a in rows:
            rx = router.reflex(t, cands, h, raw_only=True)
            if rx.get("available") and a in rx["probs"]:
                labs = list(rx["probs"])
                P.append([rx["probs"][k] for k in labs]); Y.append(labs.index(a))
        if len(P) >= MIN:
            lp = [np.log(np.clip(np.array(p), 1e-9, 1)) for p in P]

            def scaled(T):
                return [np.exp(l / T) / np.exp(l / T).sum() for l in lp]

            def nll(T):
                return -float(np.mean([math.log(max(1e-12, s[y])) for s, y in zip(scaled(T), Y)]))
            Ts = np.exp(np.linspace(math.log(0.3), math.log(5), 60))
            T = float(min(Ts, key=nll))
            before = [(max(p), int(np.argmax(p)) == y) for p, y in zip(P, Y)]
            after = [(float(max(s)), int(np.argmax(s)) == y) for s, y in zip(scaled(T), Y)]
            bb, eb = _bins(*zip(*before)); ba, ea = _bins(*zip(*after))
            res["reflex"] = {"version": h.get("version"), "T": round(T, 3), "n": len(P), "ece_before": eb, "ece_after": ea,
                             "bins_before": bb, "bins_after": ba, "accuracy": round(float(np.mean([o for _, o in before])), 3)}
        else:
            res["reflex"] = {"version": h.get("version"), "n": len(P), "skipped": f"needs {MIN} labelled sentences, has {len(P)}"}
    # reasoning model: Platt per checkpoint
    res["reasoning"] = {}
    if router.laya_state()["state"] == "ready":
        per: dict[str, list] = {}
        for t, a in rows:
            ly = router.ask_laya(t, cands)
            if ly.get("available"):
                per.setdefault(ly["model"], []).append((ly["raw_confidence"], ly["choice"] == a))
        for model, pts in per.items():
            conf, ok = [c for c, _ in pts], [o for _, o in pts]
            if len(pts) < MIN:
                res["reasoning"][model] = {"n": len(pts), "skipped": f"needs {MIN} labelled sentences in its languages, has {len(pts)}",
                                           "accuracy": round(float(np.mean(ok)), 3)}
                continue
            a, b = _platt(conf, ok)
            after = [1 / (1 + math.exp(-(a * _logit(c) + b))) for c in conf]
            bb, eb = _bins(conf, ok); ba, ea = _bins(after, ok)
            res["reasoning"][model] = {"a": round(a, 4), "b": round(b, 4), "n": len(pts), "ece_before": eb, "ece_after": ea,
                                       "bins_before": bb, "bins_after": ba, "accuracy": round(float(np.mean(ok)), 3)}
    res["audit"] = get().get("audit", [])                    # a refit keeps the on/off history
    store.write(FILE, res)
    return res


def get() -> dict:
    return store.read(FILE) or {"state": "never", "applied": False}


def set_applied(on: bool, by: str) -> dict:
    c = get()
    c["applied"] = bool(on)
    c.setdefault("audit", []).append({"at": time.time(), "by": by, "applied": bool(on)})
    store.write(FILE, c)
    return c


def start(by: str) -> None:
    if get().get("state") == "fitting" and time.time() - get().get("started", 0) < 600:
        raise RuntimeError("already fitting")
    threading.Thread(target=fit, args=(by,), daemon=True).start()


# ── used by the router when "use calibrated confidence" is on ─────────────────────────
def reflex_probs(probs: dict, version: str | None) -> dict:
    c = get()
    r = c.get("reflex") or {}
    if not c.get("applied") or not r.get("T") or r.get("version") != version or not probs:
        return probs
    l = {k: math.log(max(1e-9, v)) / r["T"] for k, v in probs.items()}
    z = sum(math.exp(v) for v in l.values())
    return {k: math.exp(v) / z for k, v in l.items()}


def reasoning_conf(conf: float, model: str) -> float:
    c = get()
    r = (c.get("reasoning") or {}).get(model) or {}
    if not c.get("applied") or "a" not in r:
        return conf
    return 1 / (1 + math.exp(-(r["a"] * _logit(conf) + r["b"])))
