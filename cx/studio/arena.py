"""Boundary arena: where two agents' territories meet, drawn so a person can see it and move it.

Points are sentences: golden sentences, human labels, and a sample of what each agent was trained
on (corpus seeds for action / human / decline, document questions for knowledge agents). Each is
embedded with the live reflex's encoder and placed on a 2-D plane:
  x  the direction from agent A's centre to agent B's centre (so the boundary is a front line)
  y  the strongest remaining spread (PCA of what is left)
The live reflex's own prediction (and how sure it is) rides along, so contested sentences stand out.
Dragging a point across the line in the UI files a human label (inbox rules apply).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

import encoder
import gates
import inbox
import router
import store

CX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = {"human": {"human_handoff"}, "decline": {"out_of_scope"}}


def _sentences(g: dict, aid: str, n: int) -> list[tuple[str, str]]:
    node = router.node(g, aid) or {}
    kind = node.get("kind")
    rng = random.Random(aid)
    out = [(x["text"], "golden") for x in gates.golden() if x["agent"] == aid]
    out += [(l["text"], "label") for l in inbox.training_rows("orchestrator") if l["agent"] == aid]
    out += [(e, "example") for e in node.get("examples", [])]
    if kind == "knowledge":
        qf = os.path.join(store.DATA, "collections", node.get("collection") or "", "questions.jsonl")
        if os.path.exists(qf):
            qs = [q for l in open(qf, encoding="utf-8") for q in json.loads(l)["questions"]]
            out += [(q, "document question") for q in rng.sample(qs, min(n, len(qs)))]
    else:
        f = os.path.join(CX, "corpus", "train_v24.jsonl")
        if os.path.exists(f):
            pool = []
            for l in open(f, encoding="utf-8"):
                o = json.loads(l)
                it = o.get("intent")
                if (kind in SEED and it in SEED[kind]) or (kind == "action" and not any(it in s for s in SEED.values())):
                    pool.append(o["text"])
            out += [(t, "training seed") for t in rng.sample(pool, min(n, len(pool)))]
    seen, uniq = set(), []
    for t, src in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append((t, src))
    return uniq[:n + 60]


def build(a: str, b: str, n: int = 90) -> dict:
    g = router.graph()
    if not router.node(g, a) or not router.node(g, b) or a == b:
        raise ValueError("pick two different agents")
    h = router.head()
    pts = [(t, src, a) for t, src in _sentences(g, a, n)] + [(t, src, b) for t, src in _sentences(g, b, n)]
    if len(pts) < 6:
        raise ValueError("not enough sentences for these two agents yet")
    enc = encoder.get(h.get("encoder") if h else None)
    X = enc.encode([t for t, _, _ in pts], max_len=64)
    lab = np.array([1 if l == b else 0 for _, _, l in pts])
    ca, cb = X[lab == 0].mean(0), X[lab == 1].mean(0)
    u = cb - ca
    u /= np.linalg.norm(u) + 1e-9
    xs = X @ u
    R = X - np.outer(xs, u)
    R -= R.mean(0)
    _, _, vt = np.linalg.svd(R, full_matrices=False)
    ys = R @ vt[0]
    xs = (xs - xs.mean()) / (np.abs(xs - xs.mean()).max() + 1e-9)
    ys = ys / (np.abs(ys).max() + 1e-9)
    pred, pb = [None] * len(pts), [None] * len(pts)
    if h and a in list(h["clf"].classes_) and b in list(h["clf"].classes_):
        P = h["clf"].predict_proba(X)
        cls = list(h["clf"].classes_)
        ia, ib = cls.index(a), cls.index(b)
        for k in range(len(pts)):
            pred[k] = cls[int(P[k].argmax())]
            pb[k] = round(float(P[k][ib] / max(1e-9, P[k][ia] + P[k][ib])), 3)
    names = {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}
    return {"a": a, "b": b, "names": names, "reflex": (h or {}).get("version"),
            "points": [{"text": t, "source": src, "label": l, "x": round(float(xs[k]), 4), "y": round(float(ys[k]), 4),
                        "pred": pred[k], "pb": pb[k], "lang": router.language(t)} for k, (t, src, l) in enumerate(pts)]}
