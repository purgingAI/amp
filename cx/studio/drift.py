"""Drift monitor: is real traffic changing under the models' feet?

Compares the last WINDOW days with the WINDOW before, on REAL customer messages (ledger sources
shadow + app; the Studio console and sandbox are people testing, not customers):
  languages      mix of en / Hinglish / Hindi / Arabic          — population stability index (PSI)
  agents         mix of where messages were routed              — PSI
  confidence     the reflex's mean confidence, and the share it was unsure about (below today's bar)
  reasoning      share that needed the slower reasoning model
  coverage       knowledge agents' median best match (are questions moving off the documents?)
  faithfulness   mean faithfulness of knowledge answers
PSI reading (the usual one): < 0.1 stable · 0.1–0.25 moving · > 0.25 shifted.
A metric is flagged when it crosses its limit; a flag is a prompt to look (Teach inbox, gap radar),
never an automatic action. Daily series for the last 2×WINDOW days ride along for sparklines.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import ledger

WINDOW = 7
SOURCES = ("shadow", "app")
MIN_N = 20
LIMITS = {"psi": 0.25, "confidence": 0.08, "unsure": 0.10, "reasoning": 0.10, "coverage": 0.05, "faith": 0.10}


def _rows(since: float) -> list[dict]:
    c = ledger._db()
    try:
        q = f"select at, source, lang, chosen, by, confidence, trace, faith_score from decisions where at >= ? and source in ({','.join('?' * len(SOURCES))})"
        return [dict(r) for r in c.execute(q, (since, *SOURCES))]
    finally:
        c.close()


def _psi(a: Counter, b: Counter) -> float | None:
    na, nb = sum(a.values()), sum(b.values())
    if na < MIN_N or nb < MIN_N:
        return None
    out = 0.0
    for k in set(a) | set(b):
        pa, pb = max(1e-4, a[k] / na), max(1e-4, b[k] / nb)
        out += (pb - pa) * math.log(pb / pa)
    return round(out, 3)


def _stats(rows: list[dict], bar: float) -> dict:
    n = len(rows)
    conf = [r["confidence"] for r in rows if r["confidence"] is not None and str(r["by"] or "").startswith(("reflex", "laya", "context"))]
    cov, laya = [], 0
    for r in rows:
        try:
            tr = json.loads(r["trace"] or "{}")
        except ValueError:
            tr = {}
        laya += bool(tr.get("laya"))
        for c in tr.get("coverage", [])[:1]:
            cov.append(c.get("best") or 0.0)
    faith = [r["faith_score"] for r in rows if r["faith_score"] is not None]
    med = sorted(cov)[len(cov) // 2] if cov else None
    return {"n": n, "langs": Counter(r["lang"] or "en" for r in rows), "agents": Counter(r["chosen"] or "—" for r in rows),
            "confidence": round(sum(conf) / len(conf), 3) if conf else None,
            "unsure": round(sum(c < bar for c in conf) / len(conf), 3) if conf else None,
            "reasoning": round(laya / n, 3) if n else None, "coverage": round(med, 3) if med is not None else None,
            "faith": round(sum(faith) / len(faith), 3) if faith else None, "faith_n": len(faith)}


def report() -> dict:
    import router
    bar = router.graph()["settings"].get("reflex_threshold", 0.5)
    now, day = time.time(), 86400
    rows = _rows(now - 2 * WINDOW * day)
    prev = [r for r in rows if r["at"] < now - WINDOW * day]
    cur = [r for r in rows if r["at"] >= now - WINDOW * day]
    a, b = _stats(prev, bar), _stats(cur, bar)
    metrics = []

    def num(key, label, better, fmt="pct"):
        x, y = a.get(key), b.get(key)
        delta = None if x is None or y is None else round(y - x, 3)
        enough = a["n"] >= MIN_N and b["n"] >= MIN_N
        worse = delta is not None and enough and ((delta < -LIMITS[key]) if better == "up" else (delta > LIMITS[key]))
        metrics.append({"key": key, "label": label, "before": x, "now": y, "delta": delta, "flag": bool(worse), "better": better, "fmt": fmt, "limit": LIMITS[key]})

    for key, label in (("langs", "Language mix"), ("agents", "Where messages go")):
        p = _psi(a[key], b[key])
        top = [k for k, _ in (b[key] + a[key]).most_common(6)]
        metrics.append({"key": key, "label": label, "psi": p, "flag": p is not None and p > LIMITS["psi"], "fmt": "psi",
                        "before": {k: round(a[key][k] / max(1, a["n"]), 3) for k in top}, "now": {k: round(b[key][k] / max(1, b["n"]), 3) for k in top}})
    num("confidence", "Reflex confidence", "up")
    num("unsure", f"Reflex unsure (below {round(bar * 100)}%)", "down")
    num("reasoning", "Needed the reasoning model", "down")
    num("coverage", "Knowledge: documents' best match", "up", "num")
    num("faith", "Knowledge: faithfulness", "up")
    # daily series for sparklines
    series = []
    for d in range(2 * WINDOW, 0, -1):
        lo, hi = now - d * day, now - (d - 1) * day
        s = _stats([r for r in rows if lo <= r["at"] < hi], bar)
        series.append({"day": round(lo), "n": s["n"], "confidence": s["confidence"], "unsure": s["unsure"], "reasoning": s["reasoning"],
                       "hinglish": round(s["langs"]["hinglish"] / s["n"], 3) if s["n"] else None})
    return {"window_days": WINDOW, "sources": list(SOURCES), "n": {"before": a["n"], "now": b["n"]}, "min_n": MIN_N,
            "metrics": metrics, "series": series, "flags": sum(m["flag"] for m in metrics), "at": now}
