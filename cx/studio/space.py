"""Embedding Space Explorer: the model's "mind" as a 3-D map a person can turn, fly through and teach.

Every point is a real sentence embedded by the LIVE reflex's encoder (384-d) and projected to 3-D by
PCA: what each agent was trained on (corpus seeds, document questions), the golden sentences, human
labels, recent real traffic from the ledger, and the live head's own misses on the blind exam.
The reflex's prediction and confidence ride along, so a sentence sitting in another agent's
constellation (a misroute) stands out — the "v24 clusters by topic, not by who should act" problem
made visible.

The decision boundary is drawn from the head itself: a voxel grid of the 3-D space is mapped back to
384-d through the PCA basis, re-normalised onto the unit sphere the head was trained on, and
classified — voxels whose neighbours disagree form the boundary shell; voxels the head is unsure
about are fog.

place(text) drops any new sentence into the same space (live traffic, or a sentence typed in).
Read-only; teaching goes through inbox.label_text as in the Arena.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter

import numpy as np

import arena
import encoder
import gates
import inbox
import ledger
import router
import store

CX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIND = os.path.join(CX, "corpus", "blind_v5.jsonl")
GRID = 16                                                  # voxels per axis
TTL = 600                                                  # seconds a build stays fresh (traffic moves)
_lock = threading.Lock()
_cur: dict = {}                                            # the last build + its PCA basis (for place)


def _agents_by_kind(g: dict, cands: list[dict]) -> dict:
    return {c.get("kind"): c["id"] for c in cands}


def _blind(g: dict, cands: list[dict]) -> list[tuple[str, str]]:
    """Blind-exam sentences with the agent that SHOULD take them (intent → agent by kind)."""
    if not os.path.exists(BLIND):
        return []
    by_kind = _agents_by_kind(g, cands)
    out = []
    for l in open(BLIND, encoding="utf-8"):
        try:
            o = json.loads(l)
        except ValueError:
            continue
        it = o.get("intent") or o.get("label")
        if not o.get("text") or not it:
            continue
        want = by_kind.get("human") if it == "human_handoff" else by_kind.get("decline") if it == "out_of_scope" else by_kind.get("action")
        if want:
            out.append((o["text"], want))
    return out


def _traffic(limit: int = 300) -> list[dict]:
    rows = []
    c = ledger._db()
    try:
        for r in c.execute("select id, at, source, text, chosen, by, confidence from decisions where at>? and source in ('app','studio','shadow') "
                           "and chosen is not null order by id desc limit ?", (time.time() - 7 * 86400, limit)):
            rows.append(dict(r))
    finally:
        c.close()
    return rows


def build(n: int = 120, refresh: bool = False) -> dict:
    with _lock:
        if _cur and not refresh and time.time() - _cur["at"] < TTL and _cur["key"] == _key():
            return dict(_cur["out"], cached=True)
        out = _build(n)
        _cur.update(out=out, at=time.time(), key=_key())
        return dict(out, cached=False)


def _key() -> str:
    h = router.head() or {}
    return f"{h.get('version')}:{len(inbox.training_rows('orchestrator'))}:{len(gates.golden())}"


def _build(n: int) -> dict:
    g = router.graph()
    h = router.head()
    if not h:
        raise ValueError("the reflex is not trained yet")
    cands = router.candidates(g)
    ids = [c["id"] for c in cands]
    names = {c["id"]: c["name"] for c in cands}
    kinds = {c["id"]: c.get("kind") for c in cands}
    pts: list[dict] = []
    for aid in ids:
        for t, src in arena._sentences(g, aid, n):
            pts.append({"text": t, "source": src, "label": aid})
    for r in _traffic():
        pts.append({"text": r["text"][:160], "source": "traffic", "label": r["chosen"] if r["chosen"] in ids else None, "src": r["source"],
                    "ledger": r["id"], "at": r["at"], "by": r["by"]})
    blind = _blind(g, cands)
    texts = [p["text"] for p in pts] + [t for t, _ in blind]
    enc = encoder.get(h.get("encoder"))
    X = enc.encode(texts, max_len=64)
    clf = h["clf"]
    cls = list(clf.classes_)
    P = clf.predict_proba(X)
    pred = [cls[int(i)] for i in P.argmax(1)]
    conf = P.max(1)
    # the blind misses join the map; the hits only count
    nb = len(pts)
    blind_hits = 0
    for k, (t, want) in enumerate(blind):
        i = nb + k
        if pred[i] == want:
            blind_hits += 1
        else:
            pts.append({"text": t, "source": "blind miss", "label": want, "_i": i})
    keep = list(range(nb)) + [p["_i"] for p in pts[nb:]]
    X, P, pred, conf = X[keep], P[keep], [pred[i] for i in keep], conf[keep]
    for p in pts:
        p.pop("_i", None)
    # ── PCA to 3-D
    mu = X.mean(0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:3].T                                           # 384 × 3
    C = Xc @ V
    scale = np.abs(C).max(0) + 1e-9
    C = C / scale
    explained = [round(float(v), 3) for v in (S[:3] ** 2 / (S ** 2).sum())]
    # ── the head's territories, as voxels of the 3-D box mapped back onto the sphere
    ax = np.linspace(-1.1, 1.1, GRID)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    G3 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    back = mu + (G3 * scale) @ V.T
    back /= np.linalg.norm(back, axis=1, keepdims=True) + 1e-9
    PG = clf.predict_proba(back.astype(np.float32))
    gcls = PG.argmax(1)
    gmax = PG.max(1)
    cls_idx = {c: i for i, c in enumerate(cls)}
    # ── per point
    points = []
    for k, p in enumerate(pts):
        probs = {ids[j]: round(float(P[k][cls_idx[ids[j]]]), 3) for j in range(len(ids)) if ids[j] in cls_idx}
        points.append(dict(p, x=round(float(C[k, 0]), 4), y=round(float(C[k, 1]), 4), z=round(float(C[k, 2]), 4),
                           pred=pred[k], conf=round(float(conf[k]), 3), probs=probs, lang=router.language(p["text"])))
    # ── constellations
    agents = []
    for aid in ids:
        idx = [k for k, p in enumerate(pts) if p["label"] == aid]
        if not idx:
            agents.append({"id": aid, "name": names[aid], "kind": kinds[aid], "n": 0, "c": [0, 0, 0], "spread": 0.3})
            continue
        cen = C[idx].mean(0)
        spread = float(np.linalg.norm(C[idx] - cen, axis=1).mean())
        agents.append({"id": aid, "name": names[aid], "kind": kinds[aid], "n": len(idx), "c": [round(float(v), 4) for v in cen], "spread": round(spread, 4),
                       "territory": round(float((gcls == cls_idx[aid]).mean()), 3) if aid in cls_idx else None})
    # ── where constellations overlap: A's sentences the head sends to B
    pairs = Counter()
    examples: dict[tuple, list] = {}
    for k, p in enumerate(pts):
        if p["label"] and pred[k] != p["label"] and p["source"] != "traffic":
            key = (p["label"], pred[k])
            pairs[key] += 1
            if len(examples.setdefault(key, [])) < 3:
                examples[key].append({"text": p["text"], "conf": round(float(conf[k]), 2), "source": p["source"]})
    contested = [{"want": a, "got": b, "n": c, "examples": examples[(a, b)]} for (a, b), c in pairs.most_common(8)]
    _cur["basis"] = {"mu": mu, "V": V, "scale": scale, "ids": ids, "cls": cls, "clf": clf, "encoder": h.get("encoder")}
    return {"reflex": h.get("version"), "encoder": os.path.basename(h.get("encoder") or "base"), "n": len(points), "explained": explained,
            "agents": agents, "points": points, "contested": contested,
            "blind": {"n": len(blind), "hits": blind_hits, "misses": len(blind) - blind_hits},
            "grid": {"k": GRID, "lo": -1.1, "hi": 1.1, "cls": [int(i) for i in gcls], "classes": cls, "conf": [round(float(v), 2) for v in gmax]},
            "at": time.time()}


def place(text: str) -> dict:
    """Drop one sentence into the current space (building it first if needed)."""
    if not _cur.get("basis"):
        build()
    b = _cur["basis"]
    text = " ".join((text or "").split())[:400]
    if not text:
        raise ValueError("type a sentence")
    v = encoder.get(b["encoder"]).encode([text], max_len=64)
    c = ((v - b["mu"]) @ b["V"]) / b["scale"]
    P = b["clf"].predict_proba(v)[0]
    cls = b["cls"]
    probs = {a: round(float(P[cls.index(a)]), 3) for a in b["ids"] if a in cls}
    best = max(probs, key=probs.get) if probs else None
    return {"text": text, "x": round(float(c[0, 0]), 4), "y": round(float(c[0, 1]), 4), "z": round(float(c[0, 2]), 4),
            "pred": best, "conf": probs.get(best), "probs": probs, "lang": router.language(text), "reflex": _cur["out"]["reflex"]}
