"""Three investor-grade views over data the Studio already has. Read-only.

boardroom()   the whole story on one screen: accuracy trajectory across every model generation, languages,
              decisions notarised, safety integrity, lessons, autonomy — every number computed live.
genome()      every reflex version as a strand of "genes": one gene per (agent × language) capability, scored on
              the 700-sentence blind exam + the golden set. Two strands side by side show exactly which genes a
              version gained or lost — mutations, not a single averaged number.
multiverse()  the last N real requests replayed through EVERY reflex version at once: where do the versions'
              futures diverge? Divergence on real traffic = the risk of switching, measured before anything ships.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict

import numpy as np

import store

LANGS = ("en", "hinglish", "hi", "ar")
_lock = threading.Lock()
_cache: dict = {}


def _cached(key: str, ttl: float, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


# ── Genome ────────────────────────────────────────────────────────────────────────────
def _heads_key() -> str:
    v = store.versions()["router"]
    return f"{len(v['list'])}:{v.get('active')}"


def genome() -> dict:
    return _cached("genome:" + _heads_key(), 3600, _genome)


def _genome() -> dict:
    import evolution
    import gates
    import router
    v = store.versions()["router"]
    live = v.get("active")
    g = router.graph()
    names = {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}
    kinds = {n["id"]: n.get("kind") for n in g["nodes"] if n["type"] == "agent"}
    strands = []
    genes_all: set = set()
    for h in v["list"]:
        try:
            head = router.load_head(h["id"])
        except Exception:                                   # noqa: BLE001
            continue
        rows, X = evolution._blind_for(head.get("encoder"))
        classes = list(head["clf"].classes_)
        keep = [i for i, (_, a) in enumerate(rows) if a in classes]
        if not keep:
            continue
        pred = head["clf"].predict(X[keep])
        cells: dict = defaultdict(lambda: [0, 0])
        for i, p in zip(keep, pred):
            t, want = rows[i]
            key = f"{want}|{router.language(t)}"
            cells[key][1] += 1
            cells[key][0] += int(p == want)
        gold, GX = evolution._golden_for(head.get("encoder"))
        gk = [i for i, x in enumerate(gold) if x["agent"] in classes]
        gp = head["clf"].predict(GX[gk]) if gk else []
        for i, p in zip(gk, gp):
            key = f"{gold[i]['agent']}|golden"
            cells[key][1] += 1
            cells[key][0] += int(p == gold[i]["agent"])
        genes = {k: {"ok": c[0], "n": c[1], "score": round(c[0] / c[1], 4)} for k, c in cells.items() if c[1]}
        genes_all |= set(genes)
        total_ok = sum(c["ok"] for k, c in genes.items() if not k.endswith("|golden"))
        total_n = sum(c["n"] for k, c in genes.items() if not k.endswith("|golden"))
        strands.append({"id": h["id"], "created": h.get("created"), "live": h["id"] == live, "challenger": bool(h.get("challenger")),
                        "parent": h.get("parent"), "recipe": h.get("recipe"), "blind": round(100 * total_ok / max(1, total_n), 2), "genes": genes})
    order_agent = sorted({k.split("|")[0] for k in genes_all}, key=lambda a: ({"action": 0, "knowledge": 1, "human": 2, "decline": 3}.get(kinds.get(a), 9), a))
    gene_order = [f"{a}|{l}" for a in order_agent for l in LANGS + ("golden",) if f"{a}|{l}" in genes_all]
    return {"live": live, "genes": [{"key": k, "agent": k.split("|")[0], "agent_name": names.get(k.split("|")[0], k.split("|")[0]),
                                     "kind": kinds.get(k.split("|")[0]), "lang": k.split("|")[1]} for k in gene_order],
            "strands": strands, "at": time.time()}


# ── Multiverse ────────────────────────────────────────────────────────────────────────
def multiverse(n: int = 120, heads: list[str] | None = None) -> dict:
    key = f"mv:{_heads_key()}:{n}:{','.join(heads or [])}:{int(time.time() // 120)}"
    return _cached(key, 120, lambda: _multiverse(n, heads))


def _multiverse(n: int, want: list[str] | None) -> dict:
    import encoder
    import ledger
    import router
    v = store.versions()["router"]
    live = v.get("active")
    g = router.graph()
    names = {x["id"]: x["name"] for x in g["nodes"] if x["type"] == "agent"}
    ids = [h["id"] for h in v["list"]]
    if want:
        pick = [h for h in want if h in ids]
    else:   # the live head, the best on the blind exam, and the newest few — a readable set of worlds
        scored = sorted([h for h in v["list"] if (h.get("scores") or {}).get("blind") is not None], key=lambda h: -h["scores"]["blind"])
        pick = list(dict.fromkeys([live] + [h["id"] for h in scored[:2]] + ids[::-1][:4]))[:6]
    c = ledger._db()
    try:
        rows = [dict(r) for r in c.execute("select id, at, source, text, lang, chosen from decisions where source in ('app','studio','shadow') "
                                           "and text is not null and length(text) > 3 order by id desc limit ?", (max(10, min(400, n)),))]
    finally:
        c.close()
    rows.reverse()
    texts = [r["text"] for r in rows]
    by_enc: dict = {}
    preds: dict = {}
    for hid in pick:
        head = router.load_head(hid)
        ek = head.get("encoder") or "base"
        if ek not in by_enc:
            by_enc[ek] = encoder.get(head.get("encoder")).encode(texts, max_len=64) if texts else np.zeros((0, 384), np.float32)
        X = by_enc[ek]
        P = head["clf"].predict_proba(X) if len(texts) else np.zeros((0, len(head["clf"].classes_)))
        cls = list(head["clf"].classes_)
        preds[hid] = [{"agent": cls[int(p.argmax())], "conf": round(float(p.max()), 3)} for p in P]
    worlds = []
    for i, r in enumerate(rows):
        outs = {h: preds[h][i] for h in pick}
        agents = Counter(o["agent"] for o in outs.values())
        worlds.append({"id": r["id"], "at": r["at"], "source": r["source"], "lang": r.get("lang") or router.language(r["text"]),
                       "text": r["text"][:120], "outcomes": outs, "split": len(agents) > 1,
                       "live_agent": outs[live]["agent"] if live in outs else None})
    pair = {}
    for h in pick:
        if h == live:
            continue
        d = sum(1 for w in worlds if w["outcomes"][h]["agent"] != w["outcomes"][live]["agent"]) if live in pick else 0
        pair[h] = {"diverge": d, "rate": round(d / max(1, len(worlds)), 4)}
    meta = {h["id"]: {"blind": (h.get("scores") or {}).get("blind"), "created": h.get("created"), "live": h["id"] == live} for h in v["list"] if h["id"] in pick}
    return {"live": live, "heads": pick, "meta": meta, "names": names, "worlds": worlds, "pairwise": pair,
            "split": sum(1 for w in worlds if w["split"]), "n": len(worlds), "at": time.time()}


# ── Boardroom ─────────────────────────────────────────────────────────────────────────
def boardroom() -> dict:
    return _cached("boardroom", 30, _boardroom)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:                                       # noqa: BLE001
        return default


def _boardroom() -> dict:
    import actions
    import ledger
    import receipts
    import router
    v = store.versions()
    rv = v["router"]
    live_head = next((h for h in rv["list"] if h["id"] == rv.get("active")), {})
    am = _safe(actions.models, {}) or {}
    enc = [m for m in am.get("models", []) if m.get("journal")]
    trajectory = [{"id": m["run"], "kind": "encoder", "score": (m.get("blind") or {}).get("score") or m.get("heldout"), "label": m.get("title"),
                   "live": bool(m.get("live")), "date": m.get("date")} for m in enc if ((m.get("blind") or {}).get("score") or m.get("heldout")) is not None]
    reflex_traj = [{"id": h["id"], "score": round(100 * h["scores"]["blind"], 1), "live": h["id"] == rv.get("active")}
                   for h in rv["list"] if (h.get("scores") or {}).get("blind") is not None]
    c = ledger._db()
    try:
        tot = c.execute("select count(*) from decisions where source in ('app','studio','shadow')").fetchone()[0]
        langs = Counter(r[0] or "en" for r in c.execute("select lang from decisions where source in ('app','studio','shadow')"))
        app_n = c.execute("select count(*) from decisions where source='app'").fetchone()[0]
        blocked = c.execute("select count(*) from decisions where by='guardrails' and chosen is null").fetchone()[0]
        p50 = c.execute("select ms_route from decisions where ms_route is not null and source in ('app','studio','shadow') order by ms_route limit 1 offset "
                        "(select count(*)/2 from decisions where ms_route is not null and source in ('app','studio','shadow'))").fetchone()
    finally:
        c.close()
    rc = _safe(lambda: receipts.listing(1), {}) or {}
    rt = _safe(lambda: __import__("redteam").snapshot(), {}) or {}
    ap = _safe(lambda: __import__("autopilot").status(), {}) or {}
    pend = _safe(lambda: __import__("recipes").pending(), {}) or {}
    labels = _safe(lambda: len(__import__("inbox").training_rows("orchestrator")), 0)
    golden = _safe(lambda: len(__import__("gates").golden()), 0)
    g = router.graph()
    agents = [{"id": n["id"], "name": n["name"], "kind": n.get("kind")} for n in g["nodes"] if n["type"] == "agent"]
    kbs = _safe(lambda: __import__("rag").collections(), []) or []
    first = trajectory[0]["score"] if trajectory else None
    return {
        "at": time.time(),
        "accuracy": {"live": round(100 * (live_head.get("scores") or {}).get("blind", 0), 1), "live_head": rv.get("active"), "encoder": am.get("live"),
                     "first": first, "trajectory": trajectory, "reflex": reflex_traj, "exam": 700},
        "traffic": {"decisions": tot, "app": app_n, "languages": dict(langs.most_common()), "p50_ms": p50[0] if p50 else None, "blocked": blocked},
        "trust": {"receipts": rc.get("total"), "integrity": rt.get("integrity"), "attacks": rt.get("armoury"), "waves": len(rt.get("waves") or []),
                  "open_breaches": len(rt.get("breaches") or []), "golden": golden},
        "learning": {"heads": len(rv["list"]), "encoders": len(enc), "labels": labels, "pending_lessons": pend.get("n"), "pinned": pend.get("pinned"),
                     "missions": len(ap.get("missions") or []), "autopilot_on": bool((ap.get("settings") or {}).get("enabled"))},
        "system": {"agents": agents, "knowledge_bases": len(kbs), "passages": sum(k.get("chunks") or 0 for k in kbs), "languages": list(LANGS)},
    }
