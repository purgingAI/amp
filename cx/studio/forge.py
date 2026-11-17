"""The Training Forge: one place to train the three things that learn — the Orchestrator's reflex,
each knowledge base's retriever, and the intent ENCODER itself (the model every reflex rides on).

snapshot()  what each of the three looks like right now: live version, champion/challenger state,
            the data available to train on, the last job — the three "anvils".
recipe(b)   validates a forge recipe for the encoder (epochs, learning rate, replay size, label share,
            contrastive term, frozen embeddings, extra corpora, whether a run without human labels
            is allowed) into a job config; estimate(cfg) says how long it will take, from the speed
            measured on this machine.

Read-only except for starting a job (app.py → jobs.start). Nothing here promotes anything.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import time

import store

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
CORPUS = os.path.join(CX, "corpus")
MODELS = os.path.join(CX, "model")
SPEED = "forge_speed.json"                                      # seconds per training step, measured
EXTRA_SETS = {"abuse_train": "abuse_train.jsonl", "brd_train": "brd_train.jsonl"}
DEFAULT_SEC_PER_STEP = 4.0


def _count(name: str) -> dict:
    """Rows, intents and language mix of one corpus file (cached per mtime: the files are large)."""
    p = os.path.join(CORPUS, name)
    if not os.path.exists(p):
        return {"rows": 0}
    key = f"{name}:{int(os.path.getmtime(p))}"
    cache = store.read("forge_corpus_cache.json", {}) or {}
    if key in cache:
        return cache[key]
    langs, intents, n = collections.Counter(), collections.Counter(), 0
    for l in open(p, encoding="utf-8"):
        try:
            o = json.loads(l)
        except ValueError:
            continue
        it = o.get("intent") or o.get("label")
        if not o.get("text") or not it:
            continue
        n += 1
        intents[it] += 1
        langs[o.get("lang") or "en*"] += 1                        # en* = unlabelled, mostly English
    out = {"rows": n, "intents": len(intents), "langs": dict(langs.most_common(6)),
           "per_intent": dict(intents.most_common()), "min_per_intent": min(intents.values()) if intents else 0}
    cache = {k: v for k, v in cache.items() if not k.startswith(name + ":")}
    cache[key] = out
    store.write("forge_corpus_cache.json", cache)
    return out


def sec_per_step() -> float:
    return float((store.read(SPEED, {}) or {}).get("sec_per_step") or DEFAULT_SEC_PER_STEP)


def recipe(b: dict, by: str) -> dict:
    """A forge recipe → the job config intent_tune reads (train_job.py). Every knob is clamped."""
    f = lambda k, d, lo, hi: max(lo, min(hi, float(b.get(k, d) if b.get(k) is not None else d)))  # noqa: E731
    extra = [k for k in (b.get("extra_sets") or []) if k in EXTRA_SETS]
    return {"kind": "intent_tune", "by": by, "challenger": True, "auto_promote": False, "threads": 4,
            "forge": True, "note": str(b.get("note") or "")[:120],
            "epochs": int(f("epochs", 1, 1, 3)), "lr": f("lr", 2e-5, 5e-6, 5e-5),
            "replay_per_intent": int(f("replay_per_intent", 120, 0, 5000)),      # 0 = the whole corpus
            "label_share": f("label_share", 0.2, 0.05, 0.5),
            "supcon": f("supcon", 0.0, 0.0, 0.5), "freeze_embeddings": bool(b.get("freeze_embeddings")),
            "extra_sets": extra, "allow_no_labels": bool(b.get("allow_no_labels")),
            "min_labels": 0 if b.get("allow_no_labels") else 5,
            "replay_corpus": "train_v24.jsonl", "reason": "forged by hand"}


def estimate(cfg: dict) -> dict:
    base = _count(cfg.get("replay_corpus", "train_v24.jsonl"))
    per = int(cfg.get("replay_per_intent", 120))
    replay = sum(min(per, n) if per else n for n in (base.get("per_intent") or {}).values())
    extra = sum(_count(EXTRA_SETS[k])["rows"] for k in cfg.get("extra_sets") or [])
    try:
        import inbox
        labels = len(inbox.training_rows("intent"))
    except Exception:                                       # noqa: BLE001
        labels = 0
    share = float(cfg.get("label_share", 0.2))
    boosted = min(labels * 4, int(share * (replay + extra) / max(0.01, 1 - share))) if labels else 0
    rows = replay + extra + boosted
    steps = int(rows / 16) * int(cfg.get("epochs", 1))
    sps = sec_per_step() * (1.35 if float(cfg.get("supcon", 0)) > 0 else 1.0) * (0.85 if cfg.get("freeze_embeddings") else 1.0)
    train = steps * sps
    exams = 2 * (700 + 95 + 407 + labels) / 64 * 1.2 + 90       # both models on every exam + loading
    return {"rows": rows, "replay": replay, "extra": extra, "labels": labels, "boosted": boosted, "steps": steps,
            "sec_per_step": round(sps, 2), "seconds": int(train + exams + 60), "seconds_max": int(train * 1.3 + exams + 120),
            "explain": f"{rows:,} sentences → {steps:,} steps × ~{sps:.1f} s, then exams for the live model and the challenger"}


def _last_job(kind: str, jobs_list: list[dict]) -> dict | None:
    return next((j for j in jobs_list if j.get("kind") == kind), None)


def snapshot() -> dict:
    import actions
    import doctor
    import inbox
    import jobs
    import rag
    import router
    jl = jobs.listing(limit=40)
    v = store.versions()
    # ── anvil 1: the reflex
    heads = v["router"].get("list", [])
    live_head = next((h for h in heads if h["id"] == v["router"].get("active")), None)
    try:
        champ = doctor.champion()
    except Exception:                                       # noqa: BLE001
        champ = {}
    reflex = {"live": v["router"].get("active"), "live_blind": ((live_head or {}).get("scores") or {}).get("blind"),
              "live_created": (live_head or {}).get("created"), "heads": len(heads), "champion": champ,
              "labels": len(inbox.training_rows("orchestrator")), "last_job": _last_job("router_tune", jl),
              "lineage": [{"id": h["id"], "blind": (h.get("scores") or {}).get("blind"), "created": h.get("created")} for h in heads[-12:]]}
    try:
        import recipes
        p = recipes.pending()
        reflex["pending"] = {"pinned": p.get("pinned"), "n": p.get("n"), "live": p.get("live")}
    except Exception:                                       # noqa: BLE001
        reflex["pending"] = None
    # ── anvil 2: retrievers, one per knowledge base
    g = router.graph()
    users = collections.defaultdict(list)
    for n in g["nodes"]:
        if n.get("type") == "agent" and n.get("collection"):
            users[n["collection"]].append(n["name"])
    kbs = [dict(c, agents=users.get(c["id"], [])) for c in rag.collections()]
    retrievers = {"kbs": kbs, "last_job": _last_job("rag_tune", jl)}
    # ── anvil 3: the encoder (intent model)
    am = actions.models()
    runs = am.get("models", [])
    live = next((m for m in runs if m.get("live")), None)
    challengers = [m for m in runs if m.get("challenger")]
    corpus = {"train_v24": _count("train_v24.jsonl"), "dev": _count("dev_multi.jsonl")["rows"], "heldout": _count("hard_eval_all.jsonl")["rows"],
              "blind": _count("blind_v5.jsonl")["rows"], "brd_eval": _count("brd_eval.jsonl")["rows"],
              "extra": {k: _count(f)["rows"] for k, f in EXTRA_SETS.items()}}
    try:
        free_gb = round(shutil.disk_usage(MODELS).free / 2**30, 1)
    except OSError:
        free_gb = None
    encoder = {"live": am.get("live"), "live_model": live, "machine": am.get("machine"), "challengers": challengers[-4:],
               "lineage": [{"run": m["run"], "blind": (m.get("blind") or {}).get("score"), "heldout": m.get("heldout"), "live": m.get("live"), "challenger": bool(m.get("challenger")), "date": m.get("date")} for m in runs if m.get("journal")],
               "labels": len(inbox.training_rows("intent")), "corpus": corpus, "disk_free_gb": free_gb, "sec_per_step": sec_per_step(),
               "last_job": _last_job("intent_tune", jl), "switch": am.get("switch"), "taxonomy": len(inbox.intent_labels())}
    return {"at": time.time(), "reflex": reflex, "retrievers": retrievers, "encoder": encoder,
            "running": jobs.running(), "jobs": jl[:14], "extra_sets": list(EXTRA_SETS)}
