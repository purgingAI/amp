"""Verified answer cache: a knowledge question asked before, answered again in milliseconds — but only
from an answer that was CHECKED.

Admission (after an answer is streamed, in the background):
  · the Claude judge read it sentence by sentence against its passages (run for every candidate, not
    just the 25% sample) and at least MIN_SCORE of it is supported, and no sentence is unsupported
  · it is not a "no information" reply
Lookup: the question (its standalone rewrite for a follow-up) is embedded with the knowledge base's
live retriever; a hit needs cosine ≥ SIM to a cached question in the SAME language — answers are
written in the customer's language.
Invalidation is built into the key: the knowledge base's documents (their ids) and its live retriever
version. Add or remove a document, or promote a retriever, and every old entry simply stops matching.
Entries expire after TTL; at most MAX per knowledge base.
"""
from __future__ import annotations

import hashlib
import threading
import time

import numpy as np

import store

FILE = "answer_cache.json"
SIM = 0.93
MIN_SCORE = 0.9
TTL = 7 * 86400
MAX = 400
_lock = threading.Lock()
_vec_cache: dict[str, np.ndarray] = {}


def _sig(cid: str) -> str:
    import rag
    m = rag.meta(cid) or {}
    docs = ",".join(sorted(d["id"] for d in m.get("docs", [])))
    return hashlib.md5(f"{docs}|{store.kb_active(cid)}".encode()).hexdigest()[:12]


def _embed(cid: str, text: str) -> np.ndarray:
    import encoder
    key = f"{store.kb_active(cid)}|{text}"
    if key not in _vec_cache:
        _vec_cache[key] = encoder.get(store.retriever_path(store.kb_active(cid))).encode([text], max_len=64)[0]
        if len(_vec_cache) > 3000:
            _vec_cache.pop(next(iter(_vec_cache)))
    return _vec_cache[key]


def _all() -> dict:
    return store.read(FILE) or {}


def lookup(cid: str, question: str, lang: str) -> dict | None:
    if not question.strip():
        return None
    d = _all().get(cid) or {}
    sig, now = _sig(cid), time.time()
    live = [e for e in d.get("entries", []) if e["sig"] == sig and e["lang"] == lang and now - e["at"] < TTL]
    if not live:
        return None
    q = _embed(cid, question)
    V = np.array([e["vec"] for e in live], dtype=np.float32)
    s = V @ q / (np.linalg.norm(V, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
    j = int(np.argmax(s))
    if s[j] < SIM:
        return None
    e = live[j]
    with _lock:
        a = _all()
        for x in a.get(cid, {}).get("entries", []):
            if x["id"] == e["id"]:
                x["hits"] = x.get("hits", 0) + 1
                x["last_hit"] = now
        store.write(FILE, a)
    return dict(e, similarity=round(float(s[j]), 4))


def admit(cid: str, question: str, lang: str, answer: str, passages: list[dict], faith_res: dict, did: int | None) -> bool:
    """Called with the faithfulness result. Returns True when the answer was cached."""
    import faith
    if not question.strip() or not answer.strip() or faith.META.search(answer):
        return False
    j = faith_res.get("judge") or {}
    if j.get("score") is None:
        return False
    if j["score"] < MIN_SCORE or any(x.get("verdict") in ("unsupported", "unknown") for x in j.get("sentences", [])):   # unsure is not verified
        return False
    vec = _embed(cid, question).astype(float).round(5).tolist()
    with _lock:
        a = _all()
        d = a.setdefault(cid, {"entries": []})
        sig = _sig(cid)
        d["entries"] = [e for e in d["entries"] if e["sig"] == sig and time.time() - e["at"] < TTL]
        if any(e["lang"] == lang and float(np.dot(e["vec"], vec) / (np.linalg.norm(e["vec"]) * np.linalg.norm(vec) + 1e-9)) >= SIM for e in d["entries"]):
            return False                                    # already have this one
        d["entries"].append({"id": hashlib.md5(f"{cid}|{lang}|{question}".encode()).hexdigest()[:10], "sig": sig, "lang": lang,
                             "question": question[:300], "answer": answer, "passages": passages[:6], "score": j["score"],
                             "vec": vec, "at": time.time(), "from_decision": did, "hits": 0})
        d["entries"] = d["entries"][-MAX:]
        store.write(FILE, a)
    return True


def status() -> dict:
    out = {}
    for cid, d in _all().items():
        sig = _sig(cid)
        live = [e for e in d.get("entries", []) if e["sig"] == sig and time.time() - e["at"] < TTL]
        out[cid] = {"entries": len(live), "stale": len(d.get("entries", [])) - len(live), "hits": sum(e.get("hits", 0) for e in live),
                    "top": sorted([{k: e[k] for k in ("question", "lang", "hits", "score", "at")} for e in live], key=lambda x: -x["hits"])[:8]}
    return {"collections": out, "sim": SIM, "min_score": MIN_SCORE, "ttl_days": TTL // 86400}


def clear(cid: str | None = None) -> None:
    with _lock:
        a = _all()
        if cid:
            a.pop(cid, None)
        else:
            a = {}
        store.write(FILE, a)
