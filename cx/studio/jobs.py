"""Training jobs: start one (as its own process), watch it, cancel it. One at a time — the laptop and
the server also serve live traffic, and two trainings on 8 cores only make both slow."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import rag
import store

HERE = os.path.dirname(os.path.abspath(__file__))


def _dir(jid: str) -> str:
    return os.path.join(store.DATA, "jobs", jid)


def state(jid: str) -> dict | None:
    s = store.read(f"jobs/{jid}/state.json")
    if s and s.get("status") == "running" and not _alive(s.get("pid")):
        s["status"] = "failed"
        s["error"] = s.get("error") or "the training process stopped unexpectedly"
    return s


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def running() -> dict | None:
    for j in listing(limit=5):
        if j.get("status") in ("running", "queued"):
            return j
    return None


def listing(limit: int = 30) -> list[dict]:
    root = os.path.join(store.DATA, "jobs")
    if not os.path.isdir(root):
        return []
    out = []
    for jid in sorted(os.listdir(root), reverse=True)[:limit]:
        s = state(jid)
        if s:
            out.append({k: s.get(k) for k in ("id", "kind", "status", "phase", "progress", "eta_s", "started", "finished",
                                              "result", "error", "config")})
    return out


def estimate(cfg: dict) -> dict:
    """A rough duration before starting — the same kind of number as "this will take 45 minutes"."""
    if cfg["kind"] == "router_tune":
        return {"seconds": 90, "explain": "embedding ~3,000 example questions and fitting the reflex"}
    if cfg["kind"] == "intent_tune":
        return {"seconds": 1500, "seconds_max": 3000, "explain": "1 epoch warm-started from the live intent model on ~5,000 replayed + labelled sentences, then exams for both models"}
    chunks = rag.all_chunks(cfg["collection"])
    n = len(chunks)
    qfile = os.path.join(store.DATA, "collections", cfg["collection"], "questions.jsonl")
    cached = sum(1 for _ in open(qfile, encoding="utf-8")) if os.path.exists(qfile) else 0
    q_sec = max(0, n - cached) / 6 * 9                                  # ~9 s per batch of 6 passages
    pairs = n * int(cfg.get("questions_per_passage", 4)) * 0.8
    steps = (pairs / 16) * int(cfg.get("epochs", 2))
    sec_step = float(store.read("speed.json", {}).get("sec_per_step", 3.0))
    rounds = int(cfg.get("max_rounds", 3))
    train = steps * sec_step
    evals = (n / 40 + 15) * (rounds + 1)
    lo = q_sec + train + evals + 60
    hi = q_sec + train * rounds * 1.15 + evals + 90
    return {"seconds": int(lo), "seconds_max": int(hi), "passages": n, "new_passages": max(0, n - cached),
            "steps_per_round": int(steps), "sec_per_step": sec_step, "rounds_max": rounds,
            "explain": f"{max(0, n - cached)} passages need questions, ~{int(steps)} training steps per round at ~{sec_step:.1f} s, "
                       f"up to {rounds} rounds if the score is below target"}


def start(cfg: dict) -> dict:
    if running():
        raise RuntimeError("a training job is already running — wait for it or cancel it")
    if cfg["kind"] == "rag_tune" and not rag.meta(cfg.get("collection", "")):
        raise ValueError("no such collection")
    jid = store.new_id("job")
    d = _dir(jid)
    os.makedirs(d, exist_ok=True)
    cfg = dict(cfg, created=time.time())
    json.dump(cfg, open(os.path.join(d, "job.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"id": jid, "kind": cfg["kind"], "status": "queued", "phase": "queued", "progress": 0, "config": cfg,
               "started": time.time()}, open(os.path.join(d, "state.json"), "w"))
    log = open(os.path.join(d, "process.log"), "ab")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "train_job.py"), d], cwd=HERE, stdout=log, stderr=log,
                         start_new_session=True, env=dict(os.environ, STUDIO_DATA=store.DATA))
    s = store.read(f"jobs/{jid}/state.json")
    s["pid"] = p.pid
    store.write(f"jobs/{jid}/state.json", s)
    if cfg.get("forge"):
        import forge
        return {"id": jid, "estimate": forge.estimate(cfg)}
    return {"id": jid, "estimate": estimate(cfg)}


def cancel(jid: str) -> bool:
    s = state(jid)
    if not s or s.get("status") not in ("running", "queued"):
        return False
    try:
        os.kill(int(s["pid"]), signal.SIGTERM)
    except (OSError, TypeError, ValueError, KeyError):
        return False
    return True


def events(jid: str, since: int = 0) -> tuple[list[dict], int]:
    p = os.path.join(_dir(jid), "events.jsonl")
    if not os.path.exists(p):
        return [], since
    with open(p, encoding="utf-8") as fh:
        fh.seek(since)
        data = fh.read()
        pos = fh.tell()
    return [json.loads(l) for l in data.splitlines() if l.strip()], pos
