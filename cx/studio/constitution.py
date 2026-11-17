"""The Constitution: rules written in plain words, enforced as tests on every version.

An ARTICLE is one rule a person can read — "Questions about refund timelines go to the refund-policy agent" —
with the structure a machine can check underneath:

    kind      must    every example sentence must reach `agent`
              never   no example sentence may reach `agent`
              block   every example must be stopped by the guardrails (before any model decides)
    agent     the agent the rule is about (must / never)
    examples  sentences in the customer's own words, in every language we serve

compile() turns a sentence of plain language into draft articles WITHOUT needing a language model: the rule is
parsed (kind, agent), then the corpora are searched by MEANING (the base encoder) for real sentences that match it,
best per language first. When a language model is available it may add drafted sentences too, marked as drafted.
Nothing counts until an admin RATIFIES the article; every change is an amendment in the log.

check() routes every example of every ratified article through the full Orchestrator with each router version —
the scorecard. gates.compare() uses violations() so a candidate may not break an article the live version keeps.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid

import numpy as np

import router
import store

FILE = "constitution.json"
KINDS = ("must", "never", "block")
CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")
POOL_SETS = ("dev_multi.jsonl", "hard_eval_all.jsonl", "brd_eval.jsonl")      # never the blind exam: it stays unseen
ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman(n: int) -> str:
    out = ""
    for v, s in ROMAN:
        while n >= v:
            out, n = out + s, n - v
    return out


# ── the document ──────────────────────────────────────────────────────────────────────
def load() -> dict:
    d = store.read(FILE) or {}
    d.setdefault("articles", [])
    d.setdefault("amendments", [])
    return d


def _save(d: dict) -> None:
    store.write(FILE, d)


def _amend(d: dict, what: str, article: dict, by: str, note: str = "") -> None:
    d["amendments"].append({"at": time.time(), "what": what, "article": article["id"], "n": article.get("n"), "text": article["text"][:200], "by": by, "note": note[:300]})
    d["amendments"] = d["amendments"][-400:]


def ratified(d: dict | None = None) -> list[dict]:
    return [a for a in (d or load())["articles"] if a.get("status") == "ratified"]


def _clean_examples(xs) -> list[dict]:
    out, seen = [], set()
    for x in xs or []:
        t = " ".join(str((x or {}).get("text") if isinstance(x, dict) else x or "").split())[:300]
        if len(t) < 3 or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append({"text": t, "lang": router.language(t), "source": str((x.get("source") if isinstance(x, dict) else "") or "manual")[:40]})
    return out[:24]


def save(art: dict, by: str) -> dict:
    """Create or edit an article. An edit of a ratified article sends it back to draft — it must be ratified again."""
    text = " ".join(str(art.get("text") or "").split())[:400]
    kind = art.get("kind")
    if len(text) < 8:
        raise ValueError("write the rule in a sentence")
    if kind not in KINDS:
        raise ValueError("kind is must, never or block")
    agent = str(art.get("agent") or "")
    if kind != "block" and not router.node(router.graph(), agent):
        raise ValueError("pick the agent this rule is about")
    ex = _clean_examples(art.get("examples"))
    if len(ex) < 2:
        raise ValueError("an article needs at least two example sentences")
    d = load()
    cur = next((a for a in d["articles"] if a["id"] == art.get("id")), None)
    if cur:
        cur.update(text=text, kind=kind, agent=agent if kind != "block" else "", examples=ex, status="draft", edited_by=by, edited_at=time.time())
        cur.pop("ratified_by", None)
        cur.pop("ratified_at", None)
        _amend(d, "amended", cur, by)
        out = cur
    else:
        out = {"id": uuid.uuid4().hex[:10], "n": max([a.get("n", 0) for a in d["articles"]] + [0]) + 1, "text": text, "kind": kind,
               "agent": agent if kind != "block" else "", "examples": ex, "status": "draft", "by": by, "at": time.time()}
        d["articles"].append(out)
        _amend(d, "proposed", out, by)
    _save(d)
    return out


def ratify(aid: str, by: str) -> dict:
    d = load()
    a = next((x for x in d["articles"] if x["id"] == aid), None)
    if not a:
        raise ValueError("no such article")
    a.update(status="ratified", ratified_by=by, ratified_at=time.time())
    _amend(d, "ratified", a, by)
    _save(d)
    return a


def repeal(aid: str, by: str, reason: str = "") -> dict:
    d = load()
    a = next((x for x in d["articles"] if x["id"] == aid), None)
    if not a:
        raise ValueError("no such article")
    a.update(status="repealed", repealed_by=by, repealed_at=time.time())
    _amend(d, "repealed", a, by, reason)
    _save(d)
    return a


# ── the compiler: plain words → draft articles ─────────────────────────────────────────
_NEVER = re.compile(r"\b(never|must\s+not|mustn'?t|should\s+not|shouldn'?t|do\s+not|don'?t|cannot|can'?t|no)\b", re.I)
_BLOCK = re.compile(r"\b(block(ed)?|stop(ped)?|refuse[ds]?|reject(ed)?|guardrails?|forbidden|not\s+allowed)\b", re.I)
_TO = re.compile(r"\b(?:go(?:es)?\s+to|must\s+reach|reach(?:es)?|are\s+(?:sent|routed|handled)\s+(?:to|by)|is\s+(?:sent|routed|handled)\s+(?:to|by)|"
                 r"handled\s+by|answered\s+by|→|->)\b", re.I)
_KIND_WORDS = {"action": "action payment payments money transfer transaction account balance personal own perform do",
               "human": "human person colleague staff hand-off handoff agent representative people support",
               "decline": "decline declined off-topic unrelated out of scope politely refuse",
               "ten": "ten small talk chat greeting hello",
               "knowledge": "knowledge documents document policy policies faq help articles rules terms"}


def _agents() -> list[dict]:
    g = router.graph()
    return [n for n in g["nodes"] if n["type"] == "agent" and n.get("enabled", True)]


def _pick_agent(clause: str, target: str) -> str:
    """The agent a clause names: words of the target part against each agent's name, description and kind."""
    words = set(re.findall(r"[a-z]+", (target or clause).lower()))
    best, score = "", 0.0
    for n in _agents():
        vocab = set(re.findall(r"[a-z]+", " ".join([n["name"], n.get("description") or "", n["id"], _KIND_WORDS.get(n.get("kind"), "")]).lower()))
        name = set(re.findall(r"[a-z]+", n["name"].lower())) - {"agent", "the"}
        s = len(words & vocab) + 2 * len(words & name)
        if s > score:
            best, score = n["id"], s
    return best


def _clauses(text: str) -> list[str]:
    parts = [p.strip(" .") for p in re.split(r";|\.\s+|\bwhile\b|\bbut\b", text) if len(p.strip(" .")) > 6]
    return parts or [text.strip()]


def _parse(clause: str) -> dict:
    kind = "block" if _BLOCK.search(clause) else "never" if _NEVER.search(clause) else "must"
    m = _TO.search(clause)
    subject, target = (clause[:m.start()], clause[m.end():]) if m else (clause, clause)
    let = re.search(r"\blet\s+(?:the\s+)?(.+?)\s+(?:handle|answer|take|deal\s+with|reply\s+to)\s+(.+)", clause, re.I)
    if kind == "never" and let:                             # "Never let the knowledge agent handle a money transfer"
        target, subject = let.group(1), let.group(2)
    if kind == "block":
        subject = clause[:_BLOCK.search(clause).start()]
    subject = re.sub(r"^\s*(?:never|always|only|when|if)\s+(?:let\s+|send\s+|route\s+)?", "", subject, flags=re.I)
    subject = re.sub(r"\s*,?\s*\b(?:they|it|are|is|must|should|be|get|gets|always)\s*$", "", _NEVER.sub("", subject).strip(" ,:-"), flags=re.I).strip(" ,:-")
    agent = "" if kind == "block" else _pick_agent(clause, target)
    return {"text": clause.strip().rstrip(".") + ".", "kind": kind, "agent": agent, "subject": subject or clause}


_BUT_NEVER = re.compile(r",?\s*(?:and\s+|but\s+)?(?:never|not)\s+(?:to\s+|by\s+)?", re.I)


def _split_parse(clause: str) -> list[dict]:
    """"X goes to Action, never to the documents" is two articles: must reach Action, and never reach the documents agent."""
    m = _TO.search(clause)
    neg = _BUT_NEVER.search(clause, m.end()) if m else None
    if not m or not neg or _NEVER.search(clause[:m.start()]):
        return [_parse(clause)]
    subject = clause[:m.start()].strip(" ,")
    pos = _parse(clause[:neg.start()])
    other = _pick_agent(clause, clause[neg.end():])
    out = [pos]
    if other and other != pos["agent"]:
        name = next((n["name"] for n in _agents() if n["id"] == other), other)
        out.append({"text": f"{subject[:1].upper() + subject[1:]} must never reach {name}.", "kind": "never", "agent": other, "subject": pos["subject"]})
    return out


_POOL: dict = {}


def _enc_key() -> str | None:
    """The live reflex's own encoder — fine-tuned on this domain; it groups intents far better than the base one,
    but it also blurs what the router itself confuses (refund policy vs my refund), so both are searched."""
    try:
        return (router.head() or {}).get("encoder")
    except Exception:                                       # noqa: BLE001
        return None


def _rows() -> list[dict]:
    """Every sentence we hold (corpora, golden set, real traffic, red-team attacks)."""
    import gates
    import ledger
    rows = []
    for f in POOL_SETS:
        p = os.path.join(CORPUS, f)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("text"):
                    rows.append({"text": r["text"], "source": f.split(".")[0]})
    rows += [{"text": x["text"], "source": "golden"} for x in gates.golden()]
    rp = os.path.join(store.DATA, "redteam_pool.jsonl")
    if os.path.exists(rp):
        rows += [{"text": json.loads(l)["text"], "source": "red team"} for l in open(rp, encoding="utf-8") if l.strip()]
    c = ledger._db()
    try:
        rows += [{"text": r[0], "source": "real traffic"} for r in c.execute(
            "select distinct text from decisions where source in ('app','studio','shadow') and text is not null and length(text) > 3 limit 3000")]
    finally:
        c.close()
    seen, uniq = set(), []
    for r in rows:
        k = " ".join(r["text"].split()).lower()
        if k not in seen and 3 < len(k) < 300:
            seen.add(k)
            uniq.append(r)
    return uniq


_POOL: dict = {}


def _embedded(rows: list[dict], ek: str | None) -> np.ndarray:
    import encoder
    key = hashlib.sha1(("\n".join(r["text"] for r in rows) + str(ek)).encode()).hexdigest()[:16]
    if key not in _POOL:
        cache = store.path(f"constitution_pool_{key}.npy")
        if os.path.exists(cache):
            X = np.load(cache)
        else:
            X = encoder.get(ek).encode([r["text"] for r in rows], max_len=64).astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            np.save(cache, X)
            old = sorted((f for f in os.listdir(store.DATA) if f.startswith("constitution_pool_")), key=lambda f: -os.path.getmtime(os.path.join(store.DATA, f)))
            for f in old[4:]:                               # the disk is small: keep the newest few embeddings only
                os.remove(os.path.join(store.DATA, f))
        _POOL[key] = X
    return _POOL[key]


def _ranked(rows: list[dict], query: list[str], ek: str | None) -> list[tuple[int, float]]:
    import encoder
    X = _embedded(rows, ek)
    q = encoder.get(ek).encode(query, max_len=64).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    sim = (X @ q.T).max(axis=1)
    order = np.argsort(-sim)[:120]
    by_lang: dict = {}
    for i in order:                                         # best per language first, then the rest
        by_lang.setdefault(router.language(rows[i]["text"]), []).append(int(i))
    out, rnd = [], 0
    while any(rnd < len(v) for v in by_lang.values()):
        out += [(v[rnd], float(sim[v[rnd]])) for v in by_lang.values() if rnd < len(v)]
        rnd += 1
    return out


def _neighbours(query: list[str], k: int = 16) -> list[dict]:
    rows = _rows()
    if not rows:
        return []
    lists = [_ranked(rows, query, None)]
    ek = _enc_key()
    if ek:
        lists.append(_ranked(rows, query, ek))
    picked, seen = [], set()
    for rank in range(max(len(l) for l in lists)):
        for li, l in enumerate(lists):
            if rank < len(l) and l[rank][0] not in seen and len(picked) < k:
                seen.add(l[rank][0])
                picked.append((l[rank][0], l[rank][1], "meaning" if li == 0 else "intent"))
    return [{"text": rows[i]["text"], "source": rows[i]["source"], "sim": round(sim, 3), "by": how, "lang": router.language(rows[i]["text"])} for i, sim, how in picked]


def _drafted(clause: dict) -> list[dict]:
    """Extra sentences from a language model, when one is available. Marked 'drafted'; a person still chooses."""
    import llm
    try:
        out = llm.complete("You write test sentences for a customer assistant's routing rules. Reply with one sentence per line, nothing else: "
                           "two in English, two in Hinglish (Hindi in Latin letters), one in Hindi (Devanagari) and one in Arabic. "
                           "Each must be something a real wallet-app customer would type that the rule clearly covers.",
                           f"Rule: {clause['text']}", max_tokens=500)
    except Exception:                                       # noqa: BLE001 — no model, no drafts: the corpus search still works
        return []
    return [{"text": l.strip(" -•\t\"'0123456789."), "source": "drafted"} for l in out.splitlines() if len(l.strip()) > 6][:6]


def compile_rule(text: str, seeds: list[str] | None = None, draft: bool = True) -> dict:
    text = " ".join((text or "").split())[:600]
    if len(text) < 8:
        raise ValueError("write the rule in a sentence")
    names = {n["id"]: n["name"] for n in _agents()}
    out = []
    for c in [c for cl in _clauses(text)[:4] for c in _split_parse(cl)][:4]:
        cands = _neighbours([c["subject"]] + [s for s in (seeds or []) if s], 16)
        if draft:
            cands = _clean_examples(_drafted(c)) + cands
        for x in cands:                                      # where the live Orchestrator sends it today
            r = router.route(x["text"])
            x["live"] = "guardrails" if r.get("by") == "guardrails" and r.get("blocked") else r.get("chosen")
        out.append(c | {"candidates": cands, "agent_name": names.get(c["agent"], "")})
    return {"clauses": out, "agents": names}


# ── the scorecard ───────────────────────────────────────────────────────────────────────
def _ok(a: dict, r: dict, g: dict) -> bool:
    if a["kind"] == "block":                                # stopped: by the guardrails, or politely declined
        return r.get("by") == "guardrails" or (router.node(g, r.get("chosen") or "") or {}).get("kind") == "decline"
    if a["kind"] == "never":
        return r.get("chosen") != a["agent"]
    return r.get("chosen") == a["agent"] and not r.get("blocked")


def heads_default() -> list[str]:
    v = store.versions()["router"]
    live = v.get("active")
    scored = sorted([h for h in v["list"] if (h.get("scores") or {}).get("blind") is not None], key=lambda h: -h["scores"]["blind"])
    return list(dict.fromkeys([live] + [h["id"] for h in scored[:2]] + [h["id"] for h in v["list"]][::-1][:3]))[:6]


def _key(arts: list[dict], heads: list[str]) -> str:
    return hashlib.sha1(json.dumps([[a["id"], a["kind"], a["agent"], [x["text"] for x in a["examples"]]] for a in arts] + [heads, router.graph().get("updated")],
                                   sort_keys=True).encode()).hexdigest()[:16]


def card() -> dict | None:
    """The last scorecard, if it is still about these articles, these versions and this flow."""
    c = store.read("constitution_card.json")
    if not c:
        return None
    return c if c.get("key") == _key(ratified(), c.get("heads") or []) else dict(c, stale=True)


def judge(a: dict, head_id: str | None, head_obj=None, g: dict | None = None) -> dict:
    g = g or router.graph()
    if head_obj is None and head_id:
        head_obj = router.load_head(head_id)
    rows = []
    for x in a["examples"]:
        r = router.route(x["text"], g, head_obj=head_obj)
        rows.append({"text": x["text"], "lang": x["lang"], "ok": _ok(a, r, g), "chosen": r.get("chosen"), "by": r.get("by"), "blocked": r.get("blocked")})
    return {"article": a["id"], "head": head_id, "n": len(rows), "kept": sum(r["ok"] for r in rows), "rows": rows}


def check(heads: list[str] | None = None):
    """Events: {type: start}, {type: cell, …} per article × version, {type: done, card}."""
    arts = ratified()
    ids = [h["id"] for h in store.versions()["router"]["list"]]
    heads = [h for h in (heads or heads_default()) if h in ids][:8]
    live = store.active("router")
    yield {"type": "start", "heads": heads, "live": live, "articles": [a["id"] for a in arts]}
    g = router.graph()
    cells = []
    for h in heads:
        ho = router.load_head(h)
        for a in arts:
            c = judge(a, h, ho, g)
            c["kept_all"] = c["kept"] == c["n"]
            cells.append(c)
            yield {"type": "cell", **c}
    per_head = {h: {"kept": sum(1 for c in cells if c["head"] == h and c["kept_all"]), "n": len(arts)} for h in heads}
    out = {"key": _key(arts, heads), "heads": heads, "live": live, "cells": cells, "per_head": per_head, "at": time.time()}
    store.write("constitution_card.json", out)
    yield {"type": "done", "card": out}


def violations(candidate_obj, live_obj, g: dict | None = None) -> dict:
    """For the court: articles the live version keeps that the candidate breaks."""
    g = g or router.graph()
    arts = ratified()
    live_kept, cand_kept, broken = 0, 0, []
    for a in arts:
        lv = judge(a, None, live_obj, g)
        cv = judge(a, None, candidate_obj, g)
        lk, ck = lv["kept"] == lv["n"], cv["kept"] == cv["n"]
        live_kept += lk
        cand_kept += ck
        if lk and not ck:
            broken.append(f"Article {roman(a['n'])}")
    return {"n": len(arts), "live": live_kept, "cand": cand_kept, "broken": broken}
