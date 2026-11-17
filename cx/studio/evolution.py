"""Model Evolution Tree & Battle: every version we ever trained, as a family tree — and a tournament.

tree()        three lineages: the intent ENCODER (journal pages, parent = the warm start), the reflex HEADS
              (versions.json, chronological chain, promotions + rollbacks from the history), and each knowledge
              base's RETRIEVERS (parent recorded at training time). Plus the change log for the ticker.
battle(kind)  a bracket of versions fighting head-to-head on the same benchmark. Reflex heads are RE-SCORED live
              on the blind exam (700 unseen sentences, encoded once with the live encoder) chunk by chunk, plus the
              golden set as the tie-break — real scoring, streamed. Encoder and retriever brackets use the scores
              recorded when they were trained (re-scoring an encoder takes minutes each).

Read-only: a champion here is a finding, not a promotion. Promotion stays in Versions, behind a court.
"""
from __future__ import annotations

import re
import time

import numpy as np

import encoder
import gates
import router
import store

_blind: dict = {}                                          # encoder path → (rows, X)


def _num(s: str) -> int:
    return int(re.sub(r"\D", "", s or "") or 0)


# ── the tree ──────────────────────────────────────────────────────────────────────────
def tree() -> dict:
    import actions
    import changelog
    import rag
    am = actions.models()
    enc_nodes = []
    runs = [m for m in am.get("models", []) if m.get("journal")]
    ids = [m["run"] for m in runs]
    for i, m in enumerate(runs):
        txt = (m.get("changed") or "") + " " + (m.get("title") or "")
        pm = re.search(r"warm start from (v\d+)", txt, re.I)
        parent = pm.group(1) if pm and pm.group(1) in ids and pm.group(1) != m["run"] else (ids[i - 1] if i else None)
        if "from scratch" in txt.lower():
            parent = None
        enc_nodes.append({"id": m["run"], "parent": parent, "date": m.get("date"), "title": m.get("title"), "changed": m.get("changed"), "lesson": m.get("lesson"),
                          "score": (m.get("blind") or {}).get("score"), "heldout": m.get("heldout"), "brd": m.get("brd"), "leaks": m.get("leaks"), "gate": m.get("gate"),
                          "live": bool(m.get("live")), "challenger": bool(m.get("challenger")), "passed": (m.get("challenger") or {}).get("passed"), "on_disk": m.get("on_disk"),
                          "by_lang": m.get("by_lang")})
    enc_events = [{"id": h.get("to"), "from": h.get("from"), "at": h.get("at"), "by": h.get("by"), "kind": "promote"} for h in am.get("history", [])]
    v = store.versions()
    heads = v["router"].get("list", [])
    hist = v["router"].get("history", [])
    rx_nodes, prev = [], None
    for h in heads:
        sc = h.get("scores") or {}
        rx_nodes.append({"id": h["id"], "parent": h.get("parent") or prev, "at": h.get("created"), "score": round(100 * sc["blind"], 1) if sc.get("blind") is not None else None,
                         "heldout": sc.get("score"), "per_agent": sc.get("blind_per_agent"), "labels": sc.get("human_labels"), "examples": h.get("examples"),
                         "encoder": (h.get("encoder") or "").split("/")[-1] or None, "job": h.get("job"), "note": h.get("note"), "live": h["id"] == v["router"].get("active"),
                         "promoted": any(x["id"] == h["id"] for x in hist)})
        prev = h["id"]
    rx_events, seen_max = [], 0
    for x in hist:
        n = _num(x["id"])
        rx_events.append({"id": x["id"], "at": x["at"], "by": x.get("by"), "kind": "rollback" if n < seen_max else "promote"})
        seen_max = max(seen_max, n)
    kbs = []
    for c in rag.collections():
        kv = store.kb_versions(c["id"])
        nodes = [{"id": x["id"], "parent": x.get("parent"), "at": x.get("created") or 0, "score": (x.get("scores") or {}).get("score"), "baseline": (x.get("baseline") or {}).get("score"),
                  "rounds": x.get("rounds"), "from_scratch": x.get("from_scratch"), "live": x["id"] == kv.get("active"), "promoted": any(y["id"] == x["id"] for y in kv.get("history", []))}
                 for x in kv.get("list", [])]
        kbs.append({"id": c["id"], "name": c["name"], "active": kv.get("active"), "nodes": nodes,
                    "events": [{"id": y["id"], "at": y["at"], "by": y.get("by"), "kind": "promote"} for y in kv.get("history", [])]})
    return {"encoder": {"live": am.get("live"), "nodes": enc_nodes, "events": enc_events},
            "reflex": {"live": v["router"].get("active"), "nodes": rx_nodes, "events": rx_events},
            "retrievers": kbs, "timeline": changelog.timeline(60)[:40], "at": time.time()}


# ── scoring a reflex head on the shared benchmark ─────────────────────────────────────
def _blind_for(enc_path: str | None):
    import doctor
    key = enc_path or "base"
    if key not in _blind:
        rows = doctor._blind_rows()
        X = encoder.get(enc_path).encode([t for t, _ in rows], max_len=64)
        _blind[key] = (rows, X)
    return _blind[key]


def _golden_for(enc_path: str | None):
    key = "golden:" + (enc_path or "base")
    if key not in _blind:
        g = gates.golden()
        X = encoder.get(enc_path).encode([x["text"] for x in g], max_len=64) if g else np.zeros((0, 384), np.float32)
        _blind[key] = (g, X)
    return _blind[key]


def score_head(ver: str, chunks: int = 7):
    """Yields cumulative blind accuracy chunk by chunk, then the final card (blind, per-agent, golden)."""
    h = router.load_head(ver)
    rows, X = _blind_for(h.get("encoder"))
    classes = list(h["clf"].classes_)
    keep = [i for i, (_, a) in enumerate(rows) if a in classes]
    n = len(keep)
    ok, per = 0, {}
    step = max(1, (n + chunks - 1) // chunks)
    for s in range(0, n, step):
        idx = keep[s:s + step]
        pred = h["clf"].predict(X[idx])
        for i, p in zip(idx, pred):
            want = rows[i][1]
            d = per.setdefault(want, {"ok": 0, "n": 0})
            d["n"] += 1
            if p == want:
                ok += 1
                d["ok"] += 1
        yield {"type": "tick", "done": min(n, s + step), "n": n, "acc": round(100 * ok / max(1, min(n, s + step)), 1)}
    g, GX = _golden_for(h.get("encoder"))
    gk = [i for i, x in enumerate(g) if x["agent"] in classes]
    gold = int(sum(1 for i, p in zip(gk, h["clf"].predict(GX[gk]) if gk else []) if p == g[i]["agent"]))
    yield {"type": "card", "blind": round(100 * ok / max(1, n), 2), "n": n, "per_agent": {k: round(100 * d["ok"] / d["n"], 1) for k, d in per.items()},
           "golden": gold, "golden_n": len(gk)}


# ── the tournament ────────────────────────────────────────────────────────────────────
def _seeds(kind: str, ids: list[str] | None, kb: str | None, t: dict) -> list[dict]:
    if kind == "reflex":
        pool = [n for n in t["reflex"]["nodes"] if n.get("score") is not None]
    elif kind == "encoder":
        pool = [n for n in t["encoder"]["nodes"] if n.get("score") is not None]
    else:
        k = next((x for x in t["retrievers"] if x["id"] == kb), None)
        pool = [n for n in (k["nodes"] if k else []) if n.get("score") is not None]
    if ids:
        pool = [n for n in pool if n["id"] in ids]
    else:
        live = [n for n in pool if n.get("live")]
        rest = sorted([n for n in pool if not n.get("live")], key=lambda n: -_num(n["id"]))
        pool = live + rest
    pool = pool[:8]
    pool.sort(key=lambda n: -(n.get("score") or 0))       # best seed first → champions meet late
    return pool


def battle(kind: str, ids: list[str] | None = None, kb: str | None = None):
    t = tree()
    seeds = _seeds(kind, ids, kb, t)
    if len(seeds) < 2:
        yield {"type": "error", "error": "need at least two scored versions to fight"}
        return
    size = 2
    while size < len(seeds):
        size *= 2
    order = [0, size - 1, size // 2, size // 2 - 1, size // 4, size - 1 - size // 4, size // 2 + size // 4, size // 2 - 1 - size // 4][:size] if size > 2 else [0, 1]
    slots = [seeds[i]["id"] if i < len(seeds) else None for i in order]
    yield {"type": "bracket", "kind": kind, "kb": kb, "size": size, "slots": slots, "seeds": [{"id": s["id"], "score": s.get("score"), "live": s.get("live")} for s in seeds],
           "benchmark": "blind exam · 700 unseen sentences, re-scored now · golden set as tie-break" if kind == "reflex" else "scores recorded when each version was trained"}
    rnd, cur = 1, slots
    live_scores: dict = {}
    stored = {s["id"]: s for s in seeds}
    while len(cur) > 1:
        nxt = []
        for m in range(0, len(cur), 2):
            a, b = cur[m], cur[m + 1]
            if a is None or b is None:
                w = a or b
                yield {"type": "match", "round": rnd, "slot": m // 2, "a": a, "b": b, "bye": True, "winner": w}
                nxt.append(w)
                continue
            yield {"type": "match", "round": rnd, "slot": m // 2, "a": a, "b": b}
            cards = {}
            for side, vid in (("a", a), ("b", b)):
                if kind == "reflex":
                    if vid in live_scores:
                        cards[side] = live_scores[vid]
                        yield {"type": "tick", "round": rnd, "slot": m // 2, "side": side, "done": cards[side]["n"], "n": cards[side]["n"], "acc": cards[side]["blind"]}
                        continue
                    try:
                        for ev in score_head(vid):
                            if ev["type"] == "tick":
                                yield {"type": "tick", "round": rnd, "slot": m // 2, "side": side, **{k: ev[k] for k in ("done", "n", "acc")}}
                            else:
                                cards[side] = ev
                    except Exception as exc:                # noqa: BLE001
                        cards[side] = {"blind": stored[vid].get("score") or 0, "n": 0, "golden": 0, "golden_n": 0, "per_agent": {}, "error": f"{type(exc).__name__}"}
                    live_scores[vid] = cards[side]
                else:
                    s = stored[vid]
                    cards[side] = {"blind": s.get("score") or 0, "golden": s.get("heldout") or s.get("baseline") or 0, "golden_n": None, "n": None, "per_agent": {}}
                    yield {"type": "tick", "round": rnd, "slot": m // 2, "side": side, "done": 1, "n": 1, "acc": cards[side]["blind"]}
            ca, cb = cards["a"], cards["b"]
            if ca["blind"] != cb["blind"]:
                w = a if ca["blind"] > cb["blind"] else b
                why = f"{max(ca['blind'], cb['blind'])} vs {min(ca['blind'], cb['blind'])} on the blind exam"
            elif (ca.get("golden") or 0) != (cb.get("golden") or 0):
                w = a if (ca.get("golden") or 0) > (cb.get("golden") or 0) else b
                why = f"tie on blind · golden {ca.get('golden')} vs {cb.get('golden')}"
            else:
                w = a if _num(a) > _num(b) else b
                why = "dead heat — the newer one advances"
            loser = b if w == a else a
            if kind == "reflex" and (cards["a" if w == a else "b"].get("golden") or 0) < (cards["b" if w == a else "a"].get("golden") or 0):
                why += f" — but it misses golden sentences {loser} gets ({cards['a' if w == a else 'b'].get('golden')} vs {cards['b' if w == a else 'a'].get('golden')}): the blind exam has no knowledge sentences; a court would catch this"
            yield {"type": "result", "round": rnd, "slot": m // 2, "a": a, "b": b, "cards": cards, "winner": w, "why": why}
            nxt.append(w)
        cur = nxt
        rnd += 1
    champ = cur[0]
    live = t["reflex"]["live"] if kind == "reflex" else t["encoder"]["live"] if kind == "encoder" else next((x["active"] for x in t["retrievers"] if x["id"] == kb), None)
    cc, lc = live_scores.get(champ) or {}, live_scores.get(live) or {}
    note = "the live version is the champion" if champ == live else f"{champ} beats the live {live} on this benchmark — hold a court before promoting"
    warn = None
    if champ != live and cc.get("golden_n") and lc.get("golden_n") and (cc.get("golden") or 0) < (lc.get("golden") or 0):
        unknown = (lc.get("golden_n") or 0) - (cc.get("golden_n") or 0)
        warn = (f"it was never trained on agents that {unknown} golden sentence(s) need, " if unknown > 0 else "") + \
               f"and gets {cc['golden']}/{cc['golden_n']} golden right vs the live head's {lc['golden']}/{lc['golden_n']} — the blind exam has no knowledge sentences, so a blind champion can still lose a court"
    yield {"type": "champion", "id": champ, "live": live, "is_live": champ == live, "card": live_scores.get(champ) or stored.get(champ), "live_card": lc or None, "note": note, "warn": warn}
