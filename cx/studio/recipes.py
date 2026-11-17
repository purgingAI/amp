"""Reflex recipes: every head is pinned to the exact data it was trained on, and a challenger is the LIVE head's
recipe plus the lessons taught since — nothing else moves.

Why: the reflex is a logistic regression refit from data each time. Before recipes, a retrain used *today's* data
(re-sampled document questions, edited examples, every label), so a challenger differed from live in many ways at
once — the court could not tell a lesson from drift (that is how h5 became irreproducible and h6–h10 decayed).

    assemble(g, cfg, labels, base)   the tagged training rows. base=None → built fresh exactly as before (same seeded
                                      sampling, so a fresh run is unchanged); base=<live recipe> → its frozen corpus
                                      seeds and document questions + TODAY's agent examples and labels = live + lessons.
    lessons(base, current)            what changed between the live recipe and now (labels, examples, agents).
    pending()                         the lessons the next challenger will learn (for the Teach view).
    save / load                       data/router/<ver>.recipe.json.gz
    reconstruct(ver)                  rebuild an un-pinned head's recipe as of its creation time and PROVE it by
                                      refitting and comparing predictions with the stored head.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict

import store

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.path.dirname(HERE), "corpus")
SEED_MAP = {"human": {"human_handoff"}, "decline": {"out_of_scope"}}
FROZEN = ("seed", "kbq")                                    # what a delta keeps from the live recipe


def hash_bucket(s: str) -> int:
    return int(hashlib.md5(s.strip().lower().encode()).hexdigest()[:6], 16) % 100


def _key(t: str) -> str:
    import inbox
    return inbox.key(t)


def _path(ver: str) -> str:
    return store.path(f"router/{ver}.recipe.json.gz")


def load(ver: str | None) -> dict | None:
    if not ver:
        return None
    p = _path(ver)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def save(ver: str, rec: dict) -> None:
    tmp = _path(ver) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False)
    os.replace(tmp, _path(ver))


# ── the rows ──────────────────────────────────────────────────────────────────────────
def _fresh_seeds(agents: list[dict], cfg: dict, rng: random.Random) -> list[list]:
    """Corpus seeds + document questions, sampled exactly as router_tune always did (rng shared, in order)."""
    kinds = {n["id"]: n.get("kind") for n in agents}
    corpus = os.path.join(CORPUS, cfg.get("seed_corpus", "train_v24.jsonl"))
    by_intent: dict[str, list[str]] = {}
    if os.path.exists(corpus):
        for line in open(corpus, encoding="utf-8"):
            o = json.loads(line)
            by_intent.setdefault(o.get("intent"), []).append(o["text"])
    rows: list[list] = []
    for aid, kind in kinds.items():
        if kind == "action":
            pool = [t for it, ts in by_intent.items() if not any(it in s for s in SEED_MAP.values()) for t in rng.sample(ts, min(40, len(ts)))]
        elif kind in SEED_MAP:
            pool = [t for it in SEED_MAP[kind] for t in by_intent.get(it, [])]
        else:
            pool = []
        rows += [[t, aid, "seed"] for t in rng.sample(pool, min(700, len(pool)))]
    for n in agents:
        rows += [[e, n["id"], "example"] for e in n.get("examples", []) if isinstance(e, str) and e.strip()]
        if n.get("kind") == "knowledge":
            qf = os.path.join(store.DATA, "collections", n.get("collection") or "", "questions.jsonl")
            if os.path.exists(qf):
                qs = [q for line in open(qf, encoding="utf-8") for q in json.loads(line)["questions"]]
                rows += [[q, n["id"], "kbq"] for q in rng.sample(qs, min(int(cfg.get("kb_questions", 40)), len(qs)))]
    return rows


def assemble(g: dict, cfg: dict, labels: list[tuple[str, str]], base: dict | None = None, log=None) -> dict:
    """→ {rows: [[text, agent, src]], lab_train, lab_exam, mode, lessons}. src ∈ seed | kbq | example | label."""
    log = log or (lambda *a, **k: None)
    agents = [n for n in g["nodes"] if n["type"] == "agent"]
    ids = [n["id"] for n in agents]
    if base is None:
        rows = _fresh_seeds(agents, cfg, random.Random(7))
        mode = "fresh"
    else:
        keep = set(ids)
        rows = [r for r in base["rows"] if r[2] in FROZEN and r[1] in keep]
        new_agents = [n for n in agents if n["id"] not in set(base.get("agents") or [])]
        if new_agents:                                      # an agent connected since live was trained: seed it fresh
            fresh = _fresh_seeds(new_agents, cfg, random.Random("new:" + ",".join(sorted(n["id"] for n in new_agents))))
            rows += [r for r in fresh if r[2] in FROZEN]
            log(f"{len(new_agents)} agent(s) new since the live head: {', '.join(n['name'] for n in new_agents)} — seeded fresh")
        for n in agents:                                    # agents' examples are teaching too: today's, not the live recipe's
            rows += [[e, n["id"], "example"] for e in n.get("examples", []) if isinstance(e, str) and e.strip()]
        mode = "delta"
    labelled = [(t, a) for t, a in labels if a in ids]
    lab_exam = [x for x in labelled if hash_bucket(x[0]) < 20]
    lab_train = [x for x in labelled if hash_bucket(x[0]) >= 20]
    if lab_train:
        boost = int(cfg.get("label_boost", 3))
        if cfg.get("balance_labels"):
            per = defaultdict(list)
            for x in lab_train:
                per[x[1]].append(x)
            cap_a = max(1, int(statistics.median([len(v) for v in per.values()]))) if per else 0
            lab_train = [x for v in per.values() for x in v[:cap_a]]
            boost = int(cfg.get("label_boost", 1))
            log(f"balanced labels: ≤{cap_a} per agent, boost ×{boost}")
        cap = max(1, int(0.25 * len(rows) / 0.75))
        rows += [[t, a, "label"] for t, a in (lab_train * max(1, boost))[:cap]]
        log(f"{len(labelled)} human labels: {len(lab_train)} train (×{max(1, boost)}, capped at 25% of the data), {len(lab_exam)} held back as an exam")
    out = {"rows": rows, "lab_train": lab_train, "lab_exam": lab_exam, "labelled": labelled, "mode": mode,
           "agents": ids, "examples": {n["id"]: [e for e in n.get("examples", []) if isinstance(e, str) and e.strip()] for n in agents}}
    out["lessons"] = lessons(base, out) if base else None
    return out


# ── what changed ──────────────────────────────────────────────────────────────────────
def lessons(base: dict | None, cur: dict) -> dict:
    if not base:
        return {"pinned": False}
    bl = {_key(t): (t, a) for t, a in base.get("labels") or []}
    cl = {_key(t): (t, a) for t, a in cur.get("labelled") or []}
    added = [{"text": t, "agent": a, "key": k} for k, (t, a) in cl.items() if k not in bl]
    removed = [{"text": t, "agent": a, "key": k} for k, (t, a) in bl.items() if k not in cl]
    moved = [{"text": t, "agent": a, "was": bl[k][1], "key": k} for k, (t, a) in cl.items() if k in bl and bl[k][1] != a]
    ex_add, ex_rem = [], []
    be, ce = base.get("examples") or {}, cur.get("examples") or {}
    for aid in set(be) | set(ce):
        b, c = set(be.get(aid, [])), set(ce.get(aid, []))
        ex_add += [{"text": t, "agent": aid} for t in sorted(c - b)]
        ex_rem += [{"text": t, "agent": aid} for t in sorted(b - c)]
    ag_add = sorted(set(cur.get("agents") or []) - set(base.get("agents") or []))
    ag_rem = sorted(set(base.get("agents") or []) - set(cur.get("agents") or []))
    n = len(added) + len(removed) + len(moved) + len(ex_add) + len(ex_rem) + len(ag_add) + len(ag_rem)
    return {"pinned": True, "base": base.get("version"), "n": n, "labels_added": added, "labels_removed": removed, "labels_moved": moved,
            "examples_added": ex_add, "examples_removed": ex_rem, "agents_added": ag_add, "agents_removed": ag_rem}


def pending() -> dict:
    """The lessons the next reflex challenger will learn = today's labels/examples/agents vs the live head's recipe."""
    import inbox
    import router
    live = store.active("router")
    base = load(live)
    if not base:
        return {"pinned": False, "live": live,
                "note": f"the live head {live} has no pinned recipe — the next challenger is rebuilt from today's data, not live + lessons"}
    g = router.graph()
    agents = [n for n in g["nodes"] if n["type"] == "agent"]
    ids = {n["id"] for n in agents}
    cur = {"labelled": [(l["text"], l["agent"]) for l in inbox.training_rows("orchestrator") if l["agent"] in ids], "agents": sorted(ids),
           "examples": {n["id"]: [e for e in n.get("examples", []) if isinstance(e, str) and e.strip()] for n in agents}}
    return dict(lessons(base, cur), live=live, verified=base.get("verified"), created=base.get("created"))


# ── a recipe for a head trained before recipes existed ────────────────────────────────
def _graph_asof(t: float) -> dict:
    import flowver
    import router
    best = None
    for v in flowver.listing(400):
        if v["at"] <= t and (not best or v["at"] > best["at"]):
            best = v
    if best:
        full = flowver.get(best["n"])
        if full and full.get("graph"):
            return full["graph"]
    return router.graph()


def reconstruct(ver: str) -> dict:
    """Rebuild <ver>'s training set as of its creation time and prove it by refitting: the refit must make the same
    predictions as the stored head on the blind exam. Saved (verified) only when it does."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import encoder
    import inbox
    import router
    v = store.versions()["router"]
    rec = next((x for x in v["list"] if x["id"] == ver), None)
    if not rec:
        raise ValueError("no such head")
    head = router.load_head(ver)
    t = rec["created"]
    g = _graph_asof(t)
    labels = [(l["text"], l["agent"]) for l in inbox.training_rows("orchestrator", until=t)]
    enc = encoder.get(head.get("encoder"))
    bf = os.path.join(CORPUS, "blind_v5.jsonl")
    blind = [json.loads(l)["text"] for l in open(bf, encoding="utf-8")] if os.path.exists(bf) else []
    Xb = enc.encode(blind, max_len=64) if blind else None
    want = head["clf"].predict(Xb) if blind else None
    tried = []
    for cfg in ({"kb_questions": 40}, {"kb_questions": 40, "balance_labels": True}, {"kb_questions": 40, "label_boost": 1},
                {"kb_questions": 40, "balance_labels": True, "label_boost": 2}):
        asm = assemble(g, cfg, labels)
        rows = [r for r in asm["rows"]]
        counts = defaultdict(int)
        for r in rows:
            counts[r[1]] += 1
        rows = [r for r in rows if counts[r[1]] >= 5]
        train = [r for r in rows if hash_bucket(r[0]) >= 20]
        X = enc.encode([r[0] for r in train], max_len=64)
        clf = LogisticRegression(C=float(rec.get("C") or 2.0), max_iter=2000, class_weight="balanced")
        clf.fit(X, [r[1] for r in train])
        agree = float((clf.predict(Xb) == want).mean()) if blind else 0.0
        tried.append({"cfg": cfg, "rows": len(rows), "agree": round(agree, 4), "labels": len(labels)})
        if agree >= 0.999:
            out = {"version": ver, "parent": None, "created": t, "cfg": cfg, "rows": rows, "labels": [list(x) for x in asm["labelled"]],
                   "agents": asm["agents"], "examples": asm["examples"], "C": rec.get("C"), "encoder": rec.get("encoder"), "mode": "reconstructed",
                   "verified": {"blind_agreement": round(agree, 4), "n": len(blind), "at": time.time()}}
            save(ver, out)
            return {"ok": True, "version": ver, "agreement": round(agree, 4), "cfg": cfg, "rows": len(rows), "labels": len(labels), "tried": tried}
    return {"ok": False, "version": ver, "tried": tried,
            "note": "no candidate recipe reproduced the head's predictions — it stays un-pinned; the next challenger will be a fresh build that pins itself"}


# ── which lesson does what: leave-one-out over the pending lessons ────────────────────
def attribute():
    """Yields: start → one `lesson` event per lesson → done. For each lesson L the challenger is refit WITHOUT L
    (everything else = live recipe + the other lessons); L's effect = full − without-L on the blind exam
    (700 unseen sentences), per agent, and on the golden set (head only), with the blind sentences L flips.
    Embeddings are computed once; each refit is a logistic regression with the live head's C."""
    import copy
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import encoder
    import gates
    import inbox
    import router
    live = store.active("router")
    base = load(live)
    if not base:
        yield {"type": "error", "error": f"the live head {live} has no pinned recipe — nothing to attribute against"}
        return
    g0 = router.graph()
    agents = [n for n in g0["nodes"] if n["type"] == "agent"]
    ids = {n["id"] for n in agents}
    labels0 = [(l["text"], l["agent"]) for l in inbox.training_rows("orchestrator") if l["agent"] in ids]
    cur = {"labelled": labels0, "agents": sorted(ids), "examples": {n["id"]: [e for e in n.get("examples", []) if isinstance(e, str) and e.strip()] for n in agents}}
    L = lessons(base, cur)
    items = [("label_added", x) for x in L["labels_added"]] + [("label_removed", x) for x in L["labels_removed"]] + \
            [("label_moved", x) for x in L["labels_moved"]] + [("example_added", x) for x in L["examples_added"]] + \
            [("example_removed", x) for x in L["examples_removed"]]
    if not items:
        yield {"type": "error", "error": "no pending lessons — a challenger now would be identical to live"}
        return
    head = router.load_head(live)
    enc = encoder.get(head.get("encoder"))
    cfg = base.get("cfg") or {}
    C = float(base.get("C") or 2.0)
    cw = cfg.get("class_weight", "balanced")
    cw = None if cw in (None, "none", "None") else cw
    # the benchmark, encoded once
    import doctor
    blind = [(t, a) for t, a in doctor._blind_rows() if a in ids]
    gold = [x for x in gates.golden() if x["agent"] in ids]
    cache: dict[str, np.ndarray] = {}

    def vecs(texts):
        miss = [t for t in dict.fromkeys(texts) if t not in cache]
        if miss:
            for t, v in zip(miss, enc.encode(miss, max_len=64)):
                cache[t] = v
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 384), np.float32)

    Xb, yb = vecs([t for t, _ in blind]), np.array([a for _, a in blind])
    Xg, yg = vecs([x["text"] for x in gold]), np.array([x["agent"] for x in gold])

    def fit(graph, labels):
        asm = assemble(graph, cfg, labels, base)
        rows = asm["rows"]
        counts: dict = {}
        for r in rows:
            counts[r[1]] = counts.get(r[1], 0) + 1
        rows = [r for r in rows if counts[r[1]] >= 5]
        train = [r for r in rows if hash_bucket(r[0]) >= 20]
        clf = LogisticRegression(C=C, max_iter=2000, class_weight=cw)
        clf.fit(vecs([r[0] for r in train]), [r[1] for r in train])
        keep_b = np.isin(yb, clf.classes_)
        keep_g = np.isin(yg, clf.classes_)
        pb = np.full(len(yb), "", dtype=object)
        pg = np.full(len(yg), "", dtype=object)
        if keep_b.any():
            pb[keep_b] = clf.predict(Xb[keep_b])
        if keep_g.any():
            pg[keep_g] = clf.predict(Xg[keep_g])
        return pb, pg

    def score(pb, pg):
        per = {a: int(((pb == a) & (yb == a)).sum()) for a in sorted(set(yb))}
        return {"blind": int((pb == yb).sum()), "blind_n": len(yb), "per_agent": per, "golden": int((pg == yg).sum()), "golden_n": len(yg)}

    lp_b = head["clf"].predict(Xb) if len(yb) else np.array([])
    lp_g = head["clf"].predict(Xg) if len(yg) else np.array([])
    full_b, full_g = fit(g0, labels0)
    S_live, S_full = score(lp_b, lp_g), score(full_b, full_g)
    yield {"type": "start", "live": live, "n": len(items), "live_score": S_live, "full_score": S_full,
           "per_agent_n": {a: int((yb == a).sum()) for a in sorted(set(yb))}}
    total = 0
    harmful = []
    for kind, x in items:
        g, labels = g0, list(labels0)
        k = x.get("key")
        if kind == "label_added":
            labels = [(t, a) for t, a in labels if _key(t) != k]
        elif kind == "label_removed":
            labels = labels + [(x["text"], x["agent"])]
        elif kind == "label_moved":
            labels = [(t, (x["was"] if _key(t) == k else a)) for t, a in labels]
        else:
            g = copy.deepcopy(g0)
            node = next((n for n in g["nodes"] if n["id"] == x["agent"]), None)
            if node is not None:
                ex = list(node.get("examples") or [])
                node["examples"] = [e for e in ex if e != x["text"]] if kind == "example_added" else ex + [x["text"]]
        wo_b, wo_g = fit(g, labels)
        S_wo = score(wo_b, wo_g)
        effect = S_full["blind"] - S_wo["blind"]
        total += effect
        if effect < 0 or (effect == 0 and S_full["golden"] - S_wo["golden"] < 0):
            harmful.append((kind, x))
        gained = [{"text": blind[i][0], "want": blind[i][1], "without": str(wo_b[i])} for i in np.where((full_b == yb) & (wo_b != yb))[0][:6]]
        lost = [{"text": blind[i][0], "want": blind[i][1], "now": str(full_b[i])} for i in np.where((full_b != yb) & (wo_b == yb))[0][:6]]
        yield {"type": "lesson", "kind": kind, "text": x["text"], "agent": x["agent"], "was": x.get("was"), "key": k,
               "effect": {"blind": effect, "golden": S_full["golden"] - S_wo["golden"],
                          "per_agent": {a: S_full["per_agent"][a] - S_wo["per_agent"].get(a, 0) for a in S_full["per_agent"]}},
               "fixes": int(((full_b == yb) & (wo_b != yb)).sum()), "breaks": int(((full_b != yb) & (wo_b == yb)).sum()),
               "gained": gained, "lost": lost}
    # the actionable answer: refit with every harmful lesson left out at once (they interact)
    without = None
    if harmful:
        g, labels = g0, list(labels0)
        for kind, x in harmful:
            k = x.get("key")
            if kind == "label_added":
                labels = [(t, a) for t, a in labels if _key(t) != k]
            elif kind == "label_removed":
                labels = labels + [(x["text"], x["agent"])]
            elif kind == "label_moved":
                labels = [(t, (x["was"] if _key(t) == k else a)) for t, a in labels]
            else:
                if g is g0:
                    g = copy.deepcopy(g0)
                node = next((n for n in g["nodes"] if n["id"] == x["agent"]), None)
                if node is not None:
                    ex = list(node.get("examples") or [])
                    node["examples"] = [e for e in ex if e != x["text"]] if kind == "example_added" else ex + [x["text"]]
        hb, hg = fit(g, labels)
        without = {"dropped": [{"kind": k_, "text": x_["text"], "agent": x_["agent"], "key": x_.get("key")} for k_, x_ in harmful], "score": score(hb, hg)}
    yield {"type": "done", "sum_of_effects": total, "total_effect": S_full["blind"] - S_live["blind"], "without_harmful": without,
           "note": "lessons interact: the effects need not add up to the total — each is measured with all the others present"}
