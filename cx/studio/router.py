"""Laya, the agent router: which agent should handle this message?

Two brains, so a new agent works the moment it is drawn on the canvas and gets better as it learns:
  reflex   a small classifier over sentence vectors, trained from each agent's example questions
           (seconds to train, versioned, scored). Knows only agents it has been trained on.
  Laya     the 421M decision model, asked a choice question over the connected agents' own
           descriptions. Needs no training, so it covers brand-new agents — and breaks ties when
           the reflex is unsure.
Before either, the guardrails (cx/engine/security.py, the same rules the live engine runs first)
can send a message straight to a person, or stop it.

The graph decides who is reachable: only agents with an enabled edge from the router are
candidates. Everything here is configuration in studio/data/graph.json, edited from the canvas.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import encoder
import store

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CX, "engine"))
sys.path.insert(0, os.path.join(CX, "docs", "model-brief"))
ENGINE = os.environ.get("STUDIO_ENGINE", "http://127.0.0.1:8440")

KINDS = {
    "action":    {"label": "Action agent", "icon": "bolt", "runs": "the live OpenWallet engine — payments, bills, balance, refunds"},
    "knowledge": {"label": "Knowledge agent", "icon": "book", "runs": "answers from an uploaded document collection"},
    "human":     {"label": "Human hand-off", "icon": "person", "runs": "hands the customer to a support colleague"},
    "decline":   {"label": "Polite decline", "icon": "shield", "runs": "kindly declines what OpenWallet does not do"},
    "ten":       {"label": "Ten (Learn guide)", "icon": "spark", "runs": "answers about our model from the Learn page notes"},
    "webhook":   {"label": "Webhook agent", "icon": "plug", "runs": "POSTs the message to your own service and returns its answer"},
}

DEFAULT_GRAPH = {
    "settings": {"reflex_threshold": 0.6, "laya_enabled": True, "laya_threshold": 0.55, "fallback": "action"},
    "nodes": [
        {"id": "in", "type": "input", "name": "Customer message", "x": 70, "y": 330},
        {"id": "guard", "type": "guard", "name": "Guardrails", "x": 300, "y": 330, "enabled": True},
        {"id": "laya", "type": "router", "name": "OpenWallet Orchestrator", "x": 560, "y": 330},
        {"id": "action", "type": "agent", "kind": "action", "name": "Action agent", "x": 900, "y": 110, "enabled": True,
         "description": "Does things in the customer's wallet: send money, pay merchants and bills, check balance, payment status, refunds, splits, reminders, recharge."},
        {"id": "knowledge", "type": "agent", "kind": "knowledge", "name": "Knowledge agent", "x": 900, "y": 250, "enabled": True,
         "collection": "openwallet-help",
         "description": "Explains how OpenWallet works from company documents: policies, fees and limits explained, KYC, how-to steps, terms and FAQs."},
        {"id": "human", "type": "agent", "kind": "human", "name": "Human hand-off", "x": 900, "y": 390, "enabled": True,
         "description": "The customer asks for a real person, is upset, or has a complaint that needs a colleague."},
        {"id": "decline", "type": "agent", "kind": "decline", "name": "Polite decline", "x": 900, "y": 530, "enabled": True,
         "description": "Anything unrelated to payments and the OpenWallet wallet: weather, jokes, general knowledge, other companies."},
        {"id": "ten", "type": "agent", "kind": "ten", "name": "Ten", "x": 900, "y": 660, "enabled": True,
         "description": "Questions about how the OpenWallet assistant's own AI model was built, trained and evaluated."},
        {"id": "out", "type": "output", "name": "Reply", "x": 1230, "y": 330},
    ],
    "edges": [
        {"from": "in", "to": "guard"}, {"from": "guard", "to": "laya"},
        {"from": "laya", "to": "action"}, {"from": "laya", "to": "knowledge"}, {"from": "laya", "to": "human"},
        {"from": "laya", "to": "decline"},
        {"from": "action", "to": "out"}, {"from": "knowledge", "to": "out"}, {"from": "human", "to": "out"},
        {"from": "decline", "to": "out"}, {"from": "ten", "to": "out"},
    ],
}


def graph() -> dict:
    g = store.read("graph.json")
    if not g:
        g = json.loads(json.dumps(DEFAULT_GRAPH))
        store.write("graph.json", g)
    return g


def _sandbox_file(user: str) -> str:
    return "sandbox/" + re.sub(r"[^a-z0-9_.-]+", "_", (user or "anon").lower())[:80] + ".json"


def sandbox_graph(user: str) -> dict:
    """A staff member's own copy of the flow (Academy): starts as a copy of the live graph."""
    g = store.read(_sandbox_file(user))
    if not g:
        g = json.loads(json.dumps(graph()))
        g["sandbox"] = user
        store.write(_sandbox_file(user), g)
    return g


def save_graph(g: dict, by: str = "", sandbox_user: str | None = None) -> dict:
    g.pop("_version", None)
    g.pop("sandbox", None)
    ids = {n["id"] for n in g.get("nodes", [])}
    for must in ("in", "laya", "out"):
        if must not in ids:
            raise ValueError(f"the graph needs its '{must}' node")
    g["edges"] = [e for e in g.get("edges", []) if e.get("from") in ids and e.get("to") in ids and e["from"] != e["to"]]
    s = g.setdefault("settings", {})
    for k, lo, hi in (("reflex_threshold", 0.0, 1.0), ("laya_threshold", 0.0, 1.0)):
        s[k] = min(hi, max(lo, float(s.get(k, DEFAULT_GRAPH["settings"][k]))))
    s["laya_enabled"] = bool(s.get("laya_enabled", True))
    g["updated"] = time.time()
    if sandbox_user:
        g["sandbox"] = sandbox_user
        store.write(_sandbox_file(sandbox_user), g)          # a sandbox never touches the live graph or its history
        return g
    store.write("graph.json", g)
    try:
        import flowver
        g["_version"] = flowver.snapshot(g, by or "unknown")
    except Exception:                                       # noqa: BLE001 — history must never block a save
        pass
    return g


def node(g: dict, nid: str) -> dict | None:
    return next((n for n in g["nodes"] if n["id"] == nid), None)


def url_ok(u: str) -> bool:
    return u.startswith("https://") or u.startswith(("http://127.0.0.1", "http://localhost"))


def readiness(n: dict) -> tuple[bool, str]:
    """Can this agent answer anything right now? An agent that cannot is never routed to."""
    kind = n.get("kind")
    if kind == "knowledge":
        import rag
        m = rag.meta(n.get("collection") or "")
        if not m:
            return False, "its document collection does not exist"
        if not sum(len(d["chunks"]) for d in m["docs"]):
            return False, "no documents yet"
    if kind == "webhook" and not url_ok(n.get("url") or ""):
        return False, "no valid URL yet"
    return True, ""


def candidates(g: dict, trace: dict | None = None) -> list[dict]:
    """Agents reachable from the router over an enabled edge, enabled, and READY to answer."""
    to = {e["to"] for e in g["edges"] if e["from"] == "laya" and e.get("enabled", True)}
    out = []
    for n in g["nodes"]:
        if n["type"] != "agent" or n["id"] not in to or not n.get("enabled", True):
            continue
        ok, why = readiness(n)
        if ok:
            out.append(n)
        elif trace is not None:
            trace.setdefault("skipped", []).append({"agent": n["id"], "why": why})
    return out


# ── the reflex ────────────────────────────────────────────────────────────────────────
def encoder_path() -> str | None:
    """The reflex reads sentences through the LIVE intent model's encoder — it was fine-tuned on our
    own customers' sentences in four languages, so its vectors already separate what people want.
    Falls back to the base multilingual model."""
    want = os.environ.get("STUDIO_ROUTER_ENCODER")
    if want:
        return want if os.path.isdir(want) else None
    m = None
    for f in ("~/amp/up.sh", "~/amp/deploy/run_server.sh"):        # laptop, then server
        try:
            m = re.search(r"LIVE_MODEL=\"?\$\{CX_LIVE_MODEL:-([\w.-]+)\}", open(os.path.expanduser(f)).read())
        except OSError:
            continue
        if m:
            break
    path = os.path.join(CX, "model", m.group(1)) if m else None
    return path if path and os.path.exists(os.path.join(path, "model.safetensors")) else None


_head = {"ver": None, "obj": None}
_head_lock = threading.Lock()


def head():
    ver = store.active("router")
    with _head_lock:
        if ver and _head["ver"] != ver:
            with open(os.path.join(store.DATA, "router", f"{ver}.pkl"), "rb") as fh:
                _head.update(ver=ver, obj=pickle.load(fh))
        return _head["obj"] if ver else None


def load_head(ver: str):
    """A specific reflex version (for gates and the version court) — never touches the live one."""
    with open(os.path.join(store.DATA, "router", f"{ver}.pkl"), "rb") as fh:
        return pickle.load(fh)


def reflex(text: str, cands: list[dict], h=None, raw_only: bool = False) -> dict:
    h = h or head()
    if not h:
        return {"available": False, "reason": "not trained yet"}
    vec = encoder.get(h.get("encoder")).encode([text], max_len=64)
    p = h["clf"].predict_proba(vec)[0]
    labels = list(h["clf"].classes_)
    known = [c["id"] for c in cands if c["id"] in labels]
    raw = {lab: float(p[i]) for i, lab in enumerate(labels)}
    z = sum(raw[k] for k in known) or 1.0
    probs = {k: raw[k] / z for k in known}
    if not raw_only:
        import calibrate
        probs = calibrate.reflex_probs(probs, h.get("version"))      # only when "use calibrated confidence" is on
    best = max(probs, key=probs.get) if probs else None
    return {"available": True, "version": h.get("version"), "probs": probs, "raw": raw, "choice": best,
            "confidence": probs.get(best, 0.0), "untrained": [c["id"] for c in cands if c["id"] not in labels]}


# ── the reasoning model (Laya, never named in the UI) ─────────────────────────────────
# Two checkpoints, chosen by the message's language — measured on the server with our own agents:
#   english       right on English, ~40% sure on everything else (it was not trained on them)
#   multilingual  Hinglish 0.91 · Hindi 0.60 · Arabic 0.92 sure and right — but wrong on English
REASONERS = {"english": ("convaiinnovations/laya", None), "multilingual": ("convaiinnovations/laya", "multilingual")}
_laya = {"state": "cold", "agents": {}, "error": ""}
_laya_lock = threading.Lock()

# Romanised Hindi: two of these in one message and it is Hinglish, not English
_HINGLISH = set("""kya kaise kitna kitne kitni kab kahan kyun kyu hai hain ho hota hoti hote tha thi mai mein me mera meri mere mujhe
muje hum humko aap apna apni ka ki ke ko se par pe nahi nahin na haan karo karna kar kardo raha rahi gaya gayi hua aaj kaisa kaisi
hui chahiye paisa paise wapas wapis aana aane aaya ayega aayega lagta lagega lagti bhai yaar abhi jaldi batao bata
samjhao bhejo bhejna dena dedo milega mila aur liye liya bhi toh wala wali wale kuch sab agar lekin phir kyunki yahan wahan
iska uska kaun kis kisko hoga hogi sakta sakti sakte chahta chahti dikhao bolo dekho theek thik accha achha matlab""".split())


def language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    words = re.findall(r"[a-z]+", text.lower())
    return "hinglish" if sum(w in _HINGLISH for w in words) >= 2 else "en"


def _mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 99.0


def laya_state() -> dict:
    loaded = sorted(_laya["agents"])
    return {"state": _laya["state"], "error": _laya["error"], "models": loaded}


def warm_laya() -> None:
    with _laya_lock:
        if _laya["state"] in ("loading", "ready"):
            return
        if _mem_available_gb() < float(os.environ.get("STUDIO_LAYA_MIN_GB", "3.0")):
            _laya.update(state="unavailable", error=f"only {_mem_available_gb():.1f} GB RAM free — the reasoning models need about 3 GB")
            return
        _laya["state"] = "loading"

    def load():
        import laya
        errs = []
        for name, (repo, sub) in REASONERS.items():
            try:
                _laya["agents"][name] = laya.load(repo, subfolder=sub) if sub else laya.load(repo)
            except Exception as exc:                        # noqa: BLE001
                errs.append(f"{name}: {type(exc).__name__}")
        _laya.update(state="ready" if _laya["agents"] else "unavailable", error="; ".join(errs)[:200])
    threading.Thread(target=load, daemon=True).start()


def ask_laya(text: str, cands: list[dict]) -> dict:
    if _laya["state"] != "ready":
        warm_laya()
        return {"available": False, "reason": _laya["error"] or f"the reasoning model is {_laya['state']}"}
    import laya
    lang = language(text)
    which = "english" if lang == "en" else "multilingual"
    agent = _laya["agents"].get(which) or next(iter(_laya["agents"].values()))
    t0 = time.time()
    q = {"route": {"type": "choice", "instructions": "Which OpenWallet agent should handle this customer message?",
                   "criteria": {c["id"]: f"{c['name']}: {c.get('description', '')}" for c in cands}}}
    out = laya.decide(agent, {"message": text}, questions=q, return_details=True)
    v = (out.values or {}).get("route") or {}
    import calibrate
    raw_conf = float(v.get("answer_confidence") or v.get("confidence") or 0.0)
    probs = v.get("probabilities") or v.get("probs") or {}
    return {"available": True, "choice": v.get("choice"), "confidence": calibrate.reasoning_conf(raw_conf, which), "raw_confidence": raw_conf,
            "probs": {k: float(x) for k, x in probs.items()} if isinstance(probs, dict) else {}, "ms": int((time.time() - t0) * 1000),
            "language": lang, "model": which}


# ── the decision ──────────────────────────────────────────────────────────────────────
GUARD_TO_HUMAN = {"human", "distress", "legal", "fraud_victim"}
GUARD_BLOCK = {"credential", "scam", "injection", "illegal", "internal", "other_person", "competitor"}


def route(text: str, g: dict | None = None, head_obj=None, ctx: dict | None = None) -> dict:
    """head_obj: score a candidate reflex version instead of the live one (gates, court, replay).
    ctx: the conversation — {"thread": [earlier turns], "engine": {frame, topic, proposal}} (context.py).
    Without it the message is judged on its own, exactly as before."""
    g = g or graph()
    t0 = time.time()
    s = g["settings"]
    trace = {"text": text, "steps": [], "language": language(text)}
    gnode = node(g, "guard")
    guard_on = gnode and gnode.get("enabled", True) and any(e["from"] == "guard" for e in g["edges"])
    if guard_on:
        try:
            import security
            verdict = security.check(text)
        except Exception as exc:                            # noqa: BLE001
            verdict = None
            trace["steps"].append({"node": "guard", "note": f"guardrails unavailable ({type(exc).__name__})"})
        trace["steps"].append({"node": "guard", "verdict": verdict or "clear"})
        if verdict in GUARD_BLOCK:
            return dict(trace, chosen=None, by="guardrails", blocked=verdict, ms=int((time.time() - t0) * 1000))
        if verdict in GUARD_TO_HUMAN and node(g, "human") and node(g, "human").get("enabled", True):
            return dict(trace, chosen="human", by="guardrails", verdict=verdict, ms=int((time.time() - t0) * 1000))
    cands = candidates(g, trace)
    if not cands:
        return dict(trace, chosen=None, by="none", error="no agent is connected to the Orchestrator", ms=int((time.time() - t0) * 1000))
    if len(cands) == 1:
        return dict(trace, chosen=cands[0]["id"], by="only-agent", ms=int((time.time() - t0) * 1000))
    rx = reflex(text, cands, head_obj)
    trace["reflex"] = rx
    if ctx is not None:
        import context
        cx = context.consider(text, rx, ctx, g, cands)
        trace["context"] = cx
        if cx.get("applied"):
            query = text
            if cx.get("standalone_needed"):
                query, how = context.standalone(text, ctx.get("thread") or [])
                cx["standalone"], cx["rewrite"] = query, how
            # a follow-up on a covered subject is judged on the stronger of its rewrite and the two turns together
            alt = " ".join([(ctx.get("thread") or [{}])[-1].get("standalone") or (ctx.get("thread") or [{}])[-1].get("text", ""), text]).strip() \
                if cx.get("standalone_needed") else None
            chosen, by = _coverage(query, g, cands, cx["agent"], "context", trace, alt=alt)
            return dict(trace, chosen=chosen, by=by, query=query, ms=int((time.time() - t0) * 1000))
    need_laya = s.get("laya_enabled", True) and (not rx.get("available") or rx.get("confidence", 0) < s["reflex_threshold"] or rx.get("untrained"))
    chosen, by = None, None
    if need_laya:
        try:
            ly = ask_laya(text, cands)
        except Exception as exc:                            # noqa: BLE001
            ly = {"available": False, "reason": f"{type(exc).__name__}"}
        trace["laya"] = ly
        new_agent_pick = ly.get("choice") in (rx.get("untrained") or []) if rx.get("available") else False
        rx_sure = rx.get("available") and rx.get("confidence", 0) >= s["reflex_threshold"]
        if rx_sure and not new_agent_pick:
            # a confident reflex keeps the agents it was trained on; the reasoning model only gets the
            # final word about agents the reflex has never seen
            pass
        elif ly.get("available") and ly.get("confidence", 0) >= s["laya_threshold"]:
            chosen, by = ly["choice"], "laya"
        elif ly.get("available") and (new_agent_pick or not rx.get("available")) and ly.get("confidence", 0) >= float(s.get("new_agent_threshold", 0.30)):
            # the reflex has never seen this agent, so its confidence says nothing about it — the
            # reasoning model's pick stands (the coverage check below still guards knowledge agents)
            chosen, by = ly["choice"], "laya"
    if not chosen and s.get("coverage_first"):
        # a flow that opts in (Genesis sandboxes): when no brain could vouch for an agent the router has never seen,
        # a knowledge agent whose documents clearly cover the question takes it
        import rag
        best_kb = None
        for c in cands:
            if c.get("kind") != "knowledge" or (rx.get("available") and c["id"] not in (rx.get("untrained") or [])):
                continue
            hits = rag.search(c.get("collection") or "", text, k=1)
            sc = hits[0]["dense"] if hits else 0.0
            if sc >= float(c.get("min_coverage", 0.30)) and (not best_kb or sc > best_kb[1]):
                best_kb = (c["id"], sc)
        if best_kb:
            chosen, by = best_kb[0], "coverage"
    if not chosen and rx.get("available") and rx.get("choice"):
        chosen, by = rx["choice"], "reflex" if rx["confidence"] >= s["reflex_threshold"] else "reflex (unsure)"
    if not chosen:
        fb = s.get("fallback")
        chosen, by = (fb if any(c["id"] == fb for c in cands) else cands[0]["id"]), "fallback"
    chosen, by = _coverage(text, g, cands, chosen, by, trace)
    return dict(trace, chosen=chosen, by=by, ms=int((time.time() - t0) * 1000))


def _ranking(trace: dict) -> list[str]:
    """Agents in the order to try for a hand-over: the trained reflex first (it knows our customers),
    the reasoning model when there is no reflex."""
    for k in ("reflex", "laya"):
        probs = (trace.get(k) or {}).get("probs") or {}
        if probs:
            return [a for a, _ in sorted(probs.items(), key=lambda x: -x[1])]
    return []


FOLLOWUP_BAR = 0.7


def _coverage(text, g, cands, chosen, by, trace, alt: str | None = None):
    """A knowledge agent must actually have something on THIS question. If the best passage in its
    documents is below the agent's minimum relevance, hand the message to the next-best agent."""
    import rag
    tried = set()
    order = _ranking(trace)
    ids = [c["id"] for c in cands]
    while True:
        n = node(g, chosen) or {}
        if n.get("kind") != "knowledge":
            return chosen, by
        tried.add(chosen)
        hits = rag.search(n.get("collection") or "", text, k=1)
        best = hits[0]["dense"] if hits else 0.0
        need = float(n.get("min_coverage", 0.30))   # measured: unrelated ~0.05, relevant ~0.33
        if alt is not None and by == "context":
            # a follow-up to this agent's own covered answer: same subject, so a lower bar (70%). The retriever
            # is brittle to small swaps ("USD" → "AED" halves the score) that do not change the subject.
            need = round(need * FOLLOWUP_BAR, 3)
            if best < need and alt:
                h2 = rag.search(n.get("collection") or "", alt, k=1)
                if h2 and h2[0]["dense"] > best:
                    hits, best = h2, h2[0]["dense"]
        trace.setdefault("coverage", []).append({"agent": chosen, "best": round(best, 3), "need": need, "followup": by == "context",
                                                 "passage": (hits[0].get("heading") or hits[0]["text"][:60]) if hits else None})
        if best >= need:
            return chosen, by
        nxt = next((a for a in order if a in ids and a not in tried), None)
        if not nxt:
            fb = g["settings"].get("fallback")
            nxt = fb if fb in ids and fb not in tried else next((a for a in ids if a not in tried), None)
        if not nxt:
            return chosen, by
        trace.setdefault("handoffs", []).append({"from": chosen, "to": nxt,
                                                 "why": f"its documents don't cover this (best match {best:.2f}, needs {need:.2f})"})
        chosen, by = nxt, "handed over"


# ── running an agent ──────────────────────────────────────────────────────────────────
BLOCK_REPLY = {
    "credential": "For your safety I can never ask for or accept your PIN, OTP or card details — please don't share them with anyone.",
    "scam": "This looks like a possible scam. Please don't send money to anyone you haven't verified — I can connect you to support if you're worried.",
    "injection": "I can only help with your OpenWallet wallet.",
    "illegal": "I can't help with that.",
    "internal": "I can't share details about OpenWallet's internal systems.",
    "other_person": "I can only show information about your own account.",
    "competitor": "I can only help with your OpenWallet wallet — I can't compare or explain other apps and services.",
}


def run_stream(nid: str, text: str, g: dict | None = None, history: list[dict] | None = None,
               query: str | None = None, session: str | None = None):
    """Events from one agent: {type: delta, text} … {type: meta, …}.
    history: earlier turns of this conversation; query: the standalone question for searching documents;
    session: a stable engine session, so the engine keeps its own context across the conversation."""
    g = g or graph()
    n = node(g, nid) or {}
    kind = n.get("kind")
    if kind == "action":
        try:
            body = json.dumps({"text": text[:500], "session": session or ("brief-studio" + os.urandom(4).hex())}).encode()
            r = urllib.request.urlopen(urllib.request.Request(ENGINE + "/api/cx/ask", data=body,
                                       headers={"Content-Type": "application/json"}), timeout=60)
            a = json.loads(r.read())
            yield {"type": "delta", "text": (a.get("say") or "(no reply)")}
            yield {"type": "meta", "intent": a.get("intent"), "confidence": a.get("confidence"), "model": a.get("model")}
        except urllib.error.HTTPError as exc:
            yield {"type": "delta", "text": "The action agent needs a signed-in customer on this server." if exc.code == 401
                   else f"The action agent answered HTTP {exc.code}."}
        except Exception as exc:                            # noqa: BLE001
            yield {"type": "delta", "text": f"The action agent (live engine) is not reachable ({type(exc).__name__})."}
    elif kind == "knowledge":
        import rag
        cid = n.get("collection") or "openwallet-help"
        if not rag.meta(cid):
            yield {"type": "delta", "text": f"This agent's collection “{cid}” has no documents yet — upload some in Knowledge."}
            return
        import answer_cache
        hit = answer_cache.lookup(cid, query or text, language(text))
        if hit:                                             # a verified answer to the same question, in this language
            yield {"type": "passages", "passages": hit["passages"]}
            yield {"type": "delta", "text": hit["answer"]}
            yield {"type": "done", "mode": "cache", "ms": 0, "cache": {"id": hit["id"], "similarity": hit["similarity"],
                                                                       "score": hit["score"], "question": hit["question"]}}
            return
        for ev in rag.answer_stream(cid, text, history=history, query=query):
            yield ev
    elif kind == "ten":
        import ten_kb
        for ev in ten_kb.ask_stream(text):
            if ev["type"] == "delta":
                yield ev
    elif kind == "human":
        yield {"type": "delta", "text": n.get("reply") or "I'll connect you with a colleague from our support team now. They'll see this conversation, so you won't need to repeat yourself."}
    elif kind == "decline":
        yield {"type": "delta", "text": n.get("reply") or "That's outside what I can help with — I'm here for your OpenWallet wallet: payments, bills, transfers, refunds and your account."}
    elif kind == "webhook":
        r = call_webhook(n, text)
        yield {"type": "delta", "text": r["answer"] if r["ok"] else f"The {n.get('name', 'webhook')} service did not answer ({r['error']})."}
        yield {"type": "meta", "webhook": {k: r[k] for k in ("status", "ms", "ok")}}
    else:
        yield {"type": "delta", "text": "This agent has no type yet."}


# ── webhook agents: any HTTP API, configured on the canvas ────────────────────────────
#   method      GET | POST | PUT
#   url         https://… (or a service on this machine); may contain {{text}} (URL-encoded)
#   body        template, default {"text": "{{text}}"} — {{text}} is JSON-escaped inside it
#   headers     kept in data/secrets.json (owner-only), never sent back to the browser
#   answer_path where the answer is in the JSON, e.g. data.message or results.0.text
#   answer_template  a sentence built from the JSON, e.g. "Your points: {{data.points}}"
#   timeout     seconds (default 15)
SECRETS = "secrets.json"


def secret_headers(aid: str) -> dict:
    return (store.read(SECRETS, {}) or {}).get(aid, {})


def set_secret_headers(aid: str, text: str) -> list[str]:
    """'Name: value' per line. An empty value keeps the stored one (the browser never sees it)."""
    allh = store.read(SECRETS, {}) or {}
    old, new = allh.get(aid, {}), {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        if not re.fullmatch(r"[A-Za-z0-9-]{1,60}", k):
            continue
        new[k] = v if v and v != "••••" else old.get(k, "")
    allh[aid] = {k: v for k, v in new.items() if v}
    store.write(SECRETS, allh)
    try:
        os.chmod(os.path.join(store.DATA, SECRETS), 0o600)
    except OSError:
        pass
    return sorted(allh[aid])


def _path(obj, path: str):
    for part in [p for p in (path or "").split(".") if p]:
        if isinstance(obj, list) and part.lstrip("-").isdigit() and -len(obj) <= int(part) < len(obj):
            obj = obj[int(part)]
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def call_webhook(n: dict, text: str) -> dict:
    from urllib.parse import quote
    url = n.get("url") or ""
    if not url_ok(url):
        return {"ok": False, "error": "no valid URL (https://… or a local service)", "status": None, "ms": 0, "raw": "", "answer": ""}
    method = (n.get("method") or "POST").upper()
    url = url.replace("{{text}}", quote(text))
    body = None
    if method != "GET":
        tpl = n.get("body") or '{"text": "{{text}}"}'
        body = tpl.replace("{{text}}", json.dumps(text)[1:-1]).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json", **secret_headers(n["id"])}
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, data=body, method=method, headers=headers),
                                   timeout=float(n.get("timeout") or 15))
        status, raw = r.status, r.read(200_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "ms": int((time.time() - t0) * 1000), "error": f"HTTP {exc.code}",
                "raw": exc.read(4000).decode("utf-8", "replace"), "answer": ""}
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "status": None, "ms": int((time.time() - t0) * 1000), "error": type(exc).__name__, "raw": "", "answer": ""}
    ms = int((time.time() - t0) * 1000)
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    answer = ""
    if data is not None and n.get("answer_template"):
        answer = re.sub(r"\{\{\s*([\w.\-]+)\s*\}\}", lambda m: "" if _path(data, m.group(1)) is None else str(_path(data, m.group(1))), n["answer_template"])
    elif data is not None and n.get("answer_path"):
        v = _path(data, n["answer_path"])
        answer = "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
    elif isinstance(data, dict):
        v = next((data[k] for k in ("answer", "text", "message", "result", "reply") if k in data), None)
        answer = v if isinstance(v, str) else json.dumps(v if v is not None else data, ensure_ascii=False)
    else:
        answer = raw
    return {"ok": True, "status": status, "ms": ms, "raw": raw[:4000], "answer": (answer or "(empty answer)")[:3000], "error": ""}
