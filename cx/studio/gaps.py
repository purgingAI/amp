"""Document-gap radar: what customers ask that a knowledge agent's documents do not answer.

A question counts when it reached (or was considered for) a knowledge agent and
  · its documents did not cover it (coverage below the agent's bar → handed on), or
  · the answer said it had no information, or
  · someone gave the answer a thumbs-down, or the faithfulness check flagged it.
Questions are grouped by topic ACROSS languages: each is compared through its English rendering
(rag.english), so "refund kitne din mein" and "how many days for a refund" land in one gap.
Each gap gets a short title and a suggested FAQ outline (Claude, cached), and is re-checked against
the knowledge base: once a new document covers it, it is marked filled.
"""
from __future__ import annotations

import hashlib
import re
import json
import time

import numpy as np

import encoder
import faith
import ledger
import llm
import rag
import router
import store

CACHE = "gaps.json"
SIM = 0.72


def _candidates(days: int = 30) -> list[dict]:
    g = router.graph()
    kn = {n["id"]: n for n in g["nodes"] if n.get("kind") == "knowledge"}
    out = []
    for r in ledger.recent(500):
        if r["at"] < time.time() - days * 86400 or r.get("source") == "sandbox":
            continue
        full = ledger.get(r["id"])
        tr = full["trace"]
        why, agent = None, None
        for c in tr.get("coverage", []):
            if c["agent"] in kn and c["best"] < c["need"]:
                why, agent = f"not covered (best match {c['best']:.2f})", c["agent"]
        if not why and full.get("chosen") in kn:
            agent = full["chosen"]
            if faith.META.search(full.get("answer") or ""):
                why = "the agent said it had no information"
            elif full.get("feedback") == -1:
                why = "thumbs-down on the answer"
            elif full.get("faith_score") is not None and full["faith_score"] < 0.67:
                why = "the answer went beyond its documents"
        if why:
            # a follow-up ("and for card payments?") is only a question together with its conversation:
            # use the standalone rewrite the Orchestrator made of it
            q = (tr.get("context") or {}).get("standalone") or tr.get("query") or r["text"]
            out.append({"id": r["id"], "text": q, "said": r["text"], "lang": r.get("lang"), "agent": agent, "why": why, "at": r["at"], "source": r["source"]})
    return out


def _title_and_outline(questions: list[str]) -> dict:
    if not llm.api_key():
        return {"title": questions[0][:60], "outline": []}
    try:
        raw = llm.complete("You help a payments company find gaps in its customer FAQ. Given customer questions its documents could "
                           "not answer, reply with: line 1 = a short topic title (max 7 words); then 3-5 lines starting with '- ' "
                           "listing what the missing FAQ section should explain. English only.",
                           "\n".join(f"- {q}" for q in questions[:12]), max_tokens=250)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return {"title": lines[0].lstrip("#- ").strip()[:80] if lines else questions[0][:60],
                "outline": [l[2:].strip() for l in lines[1:] if l.startswith("- ") and len(l[2:].split()) >= 3][:5]}
    except Exception:                                       # noqa: BLE001
        return {"title": questions[0][:60], "outline": []}


def scan() -> dict:
    cands = _candidates()
    old = (store.read(CACHE) or {}).get("labels", {})
    gaps = []
    if cands:
        texts = [rag.english(c["text"]) or c["text"] for c in cands]
        V = encoder.get(None).encode(texts, max_len=64)
        cents, members = [], []
        for i in range(len(cands)):
            best, j = -1.0, -1
            for k, cv in enumerate(cents):
                if members[k] and cands[members[k][0]]["agent"] != cands[i]["agent"]:
                    continue
                s = float(V[i] @ cv / (np.linalg.norm(cv) + 1e-9))
                if s > best:
                    best, j = s, k
            if best >= SIM:
                members[j].append(i)
                cents[j] = cents[j] + V[i]
            else:
                cents.append(V[i].copy())
                members.append([i])
        g = router.graph()
        for ms in members:
            qs = [cands[i] for i in ms]
            uniq = list(dict.fromkeys(q["text"] for q in qs))
            sig = hashlib.md5("|".join(sorted(uniq)).encode()).hexdigest()[:12]
            lab = old.get(sig) or _title_and_outline(uniq)
            old[sig] = lab
            agent = qs[0]["agent"]
            node = router.node(g, agent) or {}
            filled = None
            if node.get("collection") and rag.meta(node["collection"]):
                hits = rag.search(node["collection"], uniq[0], k=1)
                need = float(node.get("min_coverage", 0.30))
                best = hits[0]["dense"] if hits else 0.0
                filled = {"now": round(best, 3), "need": need, "filled": best >= need + 0.05}
            gaps.append({"id": sig, "title": lab["title"], "outline": lab["outline"], "agent": agent, "agent_name": node.get("name", agent),
                         "count": len(qs), "questions": uniq[:12], "langs": sorted({q["lang"] or "en" for q in qs}),
                         "why": sorted({q["why"].split(" (")[0] for q in qs}), "first": min(q["at"] for q in qs),
                         "last": max(q["at"] for q in qs), "ids": [q["id"] for q in qs][:50], "coverage": filled})
    gaps.sort(key=lambda x: (x["coverage"] or {}).get("filled", False) * 1000 - x["count"] * 10 - x["last"] / 1e9)
    out = {"at": time.time(), "gaps": gaps, "questions": len(cands), "labels": old}
    store.write(CACHE, out)
    return out


def get(refresh: bool = False) -> dict:
    c = store.read(CACHE)
    if refresh or not c or time.time() - c.get("at", 0) > 600:
        c = scan()
    return {k: v for k, v in c.items() if k != "labels"}


# ── closing a gap: a FAQ template a person fills in, published into the knowledge base ────────
FILL = "‹fill in›"


def draft(gid: str) -> dict:
    """A template, never an answer: the documents do not cover this, so nothing here can know OpenWallet's
    policy. It carries what customers actually asked and what the section should explain; every answer
    line is a blank the policy owner fills in. Publishing refuses while any blank is left."""
    g = next((x for x in get().get("gaps", []) if x["id"] == gid), None)
    if not g:
        raise ValueError("no such gap — rescan")
    qs = list(dict.fromkeys(rag.english(q) or q for q in g["questions"]))[:8]
    lines = [f"# {g['title']}", "", "_What customers asked (from the gap radar):_", ""] + [f"- {q}" for q in qs] + [""]
    for o in (g["outline"] or ["The answer customers need"]):
        lines += [f"## {o.rstrip('.')}", "", FILL, ""]
    return {"gap": gid, "agent": g["agent"], "collection": (router.node(router.graph(), g["agent"]) or {}).get("collection"),
            "name": "FAQ - " + re.sub(r"[^\w -]+", "", g["title"])[:60].strip() + ".md", "markdown": "\n".join(lines).strip() + "\n"}


def publish(gid: str, cid: str, name: str, markdown: str, by: str) -> dict:
    if FILL in markdown:
        raise ValueError("fill in every ‹fill in› line first — publishing a blank would teach the agent nothing")
    if len(re.sub(r"\s+", " ", markdown)) < 60:
        raise ValueError("that is too short to answer anyone")
    if not rag.meta(cid):
        raise ValueError("no such knowledge base")
    name = (name or "FAQ.md").strip()
    if not name.lower().endswith((".md", ".txt")):
        name += ".md"
    out = rag.add_document(cid, name, markdown.encode("utf-8"), by)
    try:
        import answer_cache
        answer_cache.clear(cid)                             # new facts: old cached answers must not outlive them
    except Exception:                                       # noqa: BLE001
        pass
    res = scan()
    g = next((x for x in res["gaps"] if x["id"] == gid), None)
    return {"ok": True, "doc": out, "gap": {k: g[k] for k in ("id", "coverage")} if g else None}
