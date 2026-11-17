"""The Live Brain: the whole decision architecture as one living picture, fed by real traffic.

Every routed decision already leaves a full trace in the ledger (guard verdict, reflex probabilities,
whether the reasoning model was consulted, context rule, knowledge coverage, hand-overs, mode).
This module compresses that trace into a small PATH the page can animate — a particle travelling
inbox → guard → orchestrator → agent → engine/reply — and serves a SNAPSHOT of the architecture:
topology, live versions, the last hour's flows, and recent decisions to catch up on.

Read-only. Nothing here changes a model, a setting or a label.
"""
from __future__ import annotations

import json
import time
from collections import Counter

import ledger
import router
import store

SOURCES = ("app", "studio", "shadow")          # sandbox practice never reaches the brain


def path(trace: dict, mode: str | None = None, cached: bool | None = None, source: str | None = None) -> dict:
    """The compact route of one decision, from its full trace (ledger `trace` or router.route result)."""
    t = trace or {}
    guard = next((s.get("verdict") for s in t.get("steps") or [] if s.get("node") == "guard" and s.get("verdict")), None)
    rx = t.get("reflex") or {}
    ly = t.get("laya") or {}
    cx = t.get("context") or {}
    probs = sorted((rx.get("probs") or {}).values(), reverse=True)
    mode = mode or ""
    out = {
        "lang": t.get("language"),
        "guard": guard,                                       # "clear", a block verdict, or None (guard off)
        "blocked": t.get("blocked"),
        "by": t.get("by"),
        "chosen": t.get("chosen"),
        "reflex": {"version": rx.get("version"), "choice": rx.get("choice"),
                   "confidence": round(float(rx.get("confidence") or 0), 3),
                   "margin": round(probs[0] - probs[1], 3) if len(probs) > 1 else None} if rx.get("available") else None,
        "reasoning": {"choice": ly.get("choice"), "confidence": round(float(ly.get("confidence") or 0), 3)} if ly.get("available") else None,
        "context": {"rule": cx.get("rule"), "why": cx.get("why")} if cx.get("applied") else None,
        "coverage": [{"agent": c.get("agent"), "best": c.get("best"), "need": c.get("need")} for c in t.get("coverage") or []],
        "handoffs": [{"from": h.get("from"), "to": h.get("to")} for h in t.get("handoffs") or []],
        "mode": mode or None,
        "engine": mode.startswith("engine"),                 # the engine carried it out (money, PIN, ledger)
        "cached": bool(cached) or mode == "cache",
        "ms": t.get("ms"),
        "source": source,
    }
    return out


def _rows(since: float, limit: int = 400) -> list[dict]:
    c = ledger._db()
    try:
        rows = c.execute("select id, at, source, text, lang, chosen, by, confidence, mode, ms_route, ms_total, trace, feedback "
                         "from decisions where at>? and source in (?,?,?) order by id desc limit ?",
                         (since, *SOURCES, max(1, min(2000, limit)))).fetchall()
    finally:
        c.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            tr = json.loads(d.pop("trace") or "{}")
        except ValueError:
            tr = {}
        d["path"] = path(tr, d.get("mode"), source=d.get("source"))
        d["text"] = (d.get("text") or "")[:90]
        out.append(d)
    return out


def recent(hours: float = 24, limit: int = 300) -> list[dict]:
    """Decisions to replay, oldest first."""
    rows = _rows(time.time() - hours * 3600, limit)
    rows.reverse()
    return rows


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    ms = sorted(r["ms_route"] for r in rows if r.get("ms_route") is not None)
    pct = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))] if ms else None  # noqa: E731
    return {"n": n,
            "by_agent": Counter((r["path"].get("blocked") and "blocked") or r.get("chosen") or "—" for r in rows),
            "by_source": Counter(r["source"] for r in rows),
            "by_lang": Counter(r.get("lang") or "en" for r in rows),
            "by_brain": Counter(r["path"].get("by") or "—" for r in rows),
            "blocked": sum(1 for r in rows if r["path"].get("blocked")),
            "reasoning": sum(1 for r in rows if r["path"].get("reasoning")),
            "cached": sum(1 for r in rows if r["path"].get("cached")),
            "engine": sum(1 for r in rows if r["path"].get("engine")),
            "handoffs": sum(len(r["path"].get("handoffs") or []) for r in rows),
            "thumbs_down": sum(1 for r in rows if (r.get("feedback") or 0) < 0),
            "ms_p50": pct(.5), "ms_p95": pct(.95)}


def _series(rows: list[dict], minutes: int = 60, step: int = 5) -> list[int]:
    """Decisions per `step` minutes over the last `minutes`, oldest bucket first."""
    now = time.time()
    buckets = [0] * (minutes // step)
    for r in rows:
        i = int((now - r["at"]) // (step * 60))
        if 0 <= i < len(buckets):
            buckets[len(buckets) - 1 - i] += 1
    return buckets


def snapshot(catchup: int = 40) -> dict:
    g = router.graph()
    v = store.versions()
    nodes = []
    for n in g["nodes"]:
        d = {k: n.get(k) for k in ("id", "type", "kind", "name", "enabled")}
        if n["type"] == "agent":
            ok, why = router.readiness(n)
            d["ready"], d["why"] = ok, why
            if n.get("kind") == "knowledge" and n.get("collection"):
                cid = n["collection"]
                meta = store.read(f"collections/{cid}/meta.json") or {}
                kv = store.kb_versions(cid)
                d["collection"] = cid
                d["kb"] = {"name": meta.get("name"), "docs": len(meta.get("docs") or []),
                           "chunks": sum(len(x.get("chunks") or []) for x in meta.get("docs") or []), "retriever": kv.get("active")}
                d["min_coverage"] = n.get("min_coverage", 0.30)
        nodes.append(d)
    hour = _rows(time.time() - 3600, 1500)
    day = _rows(time.time() - 86400, 2000)
    try:
        import actions
        intent = actions.live_run()
    except Exception:                                       # noqa: BLE001
        intent = None
    try:
        import learning
        rollout = learning.settings()["rollout"]["percent"]
    except Exception:                                       # noqa: BLE001
        rollout = None
    try:
        import calibrate
        calibrated = bool((calibrate.get() or {}).get("applied"))
    except Exception:                                       # noqa: BLE001
        calibrated = False
    drift = None
    try:
        import drift as _drift
        rep = _drift.report()
        drift = {"flags": rep.get("flags", 0), "n": rep.get("n"),
                 "flagged": [m["label"] for m in rep.get("metrics", []) if m.get("flag")],
                 "agents": (next((m for m in rep.get("metrics", []) if m["key"] == "agents"), {}) or {})}
    except Exception:                                       # noqa: BLE001
        pass
    return {
        "at": time.time(),
        "topology": {"nodes": nodes, "edges": g["edges"], "settings": g["settings"]},
        "versions": {"reflex": v["router"]["active"], "intent": intent, "calibrated": calibrated,
                     "laya": (router.laya_state() or {}).get("state")},
        "rollout": rollout,
        "hour": _stats(hour), "day": _stats(day),
        "series": _series(day, 60, 5),
        "recent": list(reversed(day[:max(0, min(200, catchup))])),          # oldest first, to catch up on
        "drift": drift,
    }
