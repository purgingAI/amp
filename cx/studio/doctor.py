"""The Label Doctor — keeps the Teach-me labels healthy so retraining makes the reflex better, not worse.

A reflex (the fast classifier) is retrained on the intent corpus + each agent's examples + the human labels from
the Teach-me inbox, and labels are boosted ×3 and sit right on the decision boundary — so a handful of bad labels
can tilt it and quietly erode one agent across every retrain (exactly how h5 → h9 drifted down on `action`).

This module reads the label set and flags the labels that hurt, so a person can drop them before retraining:

  bare-entity   one or two words, likely a name or a menu word ("Alex", "Merchant") — ambiguous out of context,
                and teaching a name → an agent drags every sentence with a name that way
  placeholder   UI text, not a customer message ("Type your request")
  fragment      no real words / too short to carry a decision
  duplicate     near-identical to another label for the same agent (over-weights one phrasing)
  conflict      the same sentence labelled to two different agents (frozen until resolved)
  imbalance     an agent has far more labels than its real share of traffic (skews the boundary toward it)

It also surfaces the champion-vs-recent picture (is the live head still the best?) so people stop chasing
retrains that lose accuracy.
"""
from __future__ import annotations

import os
import re

import encoder
import inbox
import router
import store

_WORD = re.compile(r"[^\W\d_]{3,}", re.U)
_PLACEHOLDER = re.compile(r"(?i)\b(type your|enter your|your request|placeholder|search\.\.\.|example|lorem|untitled|click here|select an?)\b")
_TOKEN = re.compile(r"[^\W_]+", re.U)
_INTERJ = set("lol haha hahaha ok okay hi hey hello hmm yo sup thanks thx bye lmao rofl yes no yeah nope cool nice wow oops".split())


def _tokens(t: str) -> list[str]:
    return _TOKEN.findall((t or "").lower())


def _blind_share() -> dict:
    """Each agent's share of the live reflex's blind set — the real traffic mix to balance labels against."""
    try:
        rl = store.versions()["router"]
        act = rl.get("active")
        entry = next((x for x in rl.get("list", []) if x["id"] == act), {})
        bpa = (entry.get("scores") or {}).get("blind_per_agent") or {}
        tot = 0
        counts = {}
        for a, v in bpa.items():
            n = int(str(v).split("/")[-1]) if "/" in str(v) else 0
            counts[a] = n
            tot += n
        return {a: round(n / tot, 3) for a, n in counts.items()} if tot else {}
    except Exception:                                       # noqa: BLE001
        return {}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def health() -> dict:
    """The label set, scored for problems. Nothing is changed."""
    g = router.graph()
    names = {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}
    labels = inbox._labels()
    conflicts = {c["key"] for c in inbox.conflicts_list()}
    by_agent: dict[str, list[dict]] = {}
    for l in labels:
        by_agent.setdefault(l.get("agent") or l.get("intent") or "?", []).append(l)

    flags = []
    # near-duplicate detection within each agent
    dup_keys = set()
    for a, ls in by_agent.items():
        toks = [(_tokens(x["text"]), x) for x in ls]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                if toks[i][0] and _jaccard(toks[i][0], toks[j][0]) >= 0.6:
                    dup_keys.add(id(toks[j][1]))             # keep the first, flag the later ones

    share_lbl = {a: len(ls) / max(1, len(labels)) for a, ls in by_agent.items()}
    share_traffic = _blind_share()

    for l in labels:
        text = l.get("text") or ""
        toks = _tokens(text)
        a = l.get("agent") or l.get("intent") or "?"
        issues = []
        if not _WORD.search(text):
            issues.append("fragment")
        if _PLACEHOLDER.search(text):
            issues.append("placeholder")
        # a bare name/menu word teaches a context-free token → an agent (e.g. a payee name → decline): harmful.
        # short interjections ("Lol", "Hi") are fine for decline, so only flag capitalised proper-noun-like tokens.
        elif len(toks) <= 2 and text.strip()[:1].isupper() and (toks[0] if toks else "") not in _INTERJ:
            issues.append("bare-entity")
        if id(l) in dup_keys:
            issues.append("duplicate")
        if l["key"] in conflicts:
            issues.append("conflict")
        if issues:
            flags.append({"key": l["key"], "text": text, "agent": a, "agent_name": names.get(a, a),
                          "lang": l.get("lang"), "by": l.get("by"), "issues": issues,
                          "severity": "high" if ({"bare-entity", "placeholder", "conflict"} & set(issues)) else "medium"})

    balance = []
    for a, ls in sorted(by_agent.items(), key=lambda kv: -len(kv[1])):
        ls_share, tr = share_lbl.get(a, 0), share_traffic.get(a)
        mark = None
        if tr is not None:
            if ls_share > tr + 0.25:
                mark = "over"                                # far more labels than this agent's real traffic
            elif tr - ls_share > 0.25:
                mark = "under"                               # under-taught relative to its traffic
        balance.append({"agent": a, "agent_name": names.get(a, a), "labels": len(ls),
                        "label_share": round(ls_share, 3), "traffic_share": tr, "flag": mark})

    return {"total": len(labels), "flagged": len(flags), "flags": flags, "balance": balance,
            "champion": champion(), "agents": names}


def _blind_rows():
    """The blind exam (blind_v5), each sentence mapped to the agent it SHOULD reach — the yardstick the court
    judges on. Read for diagnosis only; never trained on."""
    import json as _json
    bf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "blind_v5.jsonl")
    if not os.path.exists(bf):
        return []
    g = router.graph()
    kinds = {n["id"]: n.get("kind") for n in g["nodes"] if n["type"] == "agent"}
    seed = {"human": {"human_handoff"}, "decline": {"out_of_scope"}}

    def to_agent(it):
        for aid, kind in kinds.items():
            if kind in seed and it in seed[kind]:
                return aid
        return next((aid for aid, k in kinds.items() if k == "action"), None)
    out = []
    for line in open(bf, encoding="utf-8"):
        r = _json.loads(line)
        out.append((r["text"], to_agent(r.get("intent") or r.get("label"))))
    return out


def _blind_keys() -> set:
    return {inbox.key(t) for t, _ in _blind_rows()}


def weakspots(limit: int = 8) -> dict:
    """Where the LIVE reflex gets the blind set wrong, grouped by want→got and language — the real patterns to
    fix. Users add fresh examples of these patterns (not the exam sentences) so the next retrain beats the champion."""
    import numpy as np
    head = router.head()
    if not head:
        return {"groups": [], "total": 0, "note": "no trained reflex yet"}
    rows = _blind_rows()
    classes = set(head["clf"].classes_)
    rows = [(t, a) for t, a in rows if a in classes]
    if not rows:
        return {"groups": [], "total": 0}
    enc = encoder.get(head.get("encoder"))
    pred = head["clf"].predict(enc.encode([t for t, _ in rows], max_len=64))
    g = router.graph()
    names = {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}
    groups: dict = {}
    miss = 0
    for (t, want), got in zip(rows, pred):
        if got == want:
            continue
        miss += 1
        lang = router.language(t)
        k = (want, got, lang)
        gp = groups.setdefault(k, {"want": want, "want_name": names.get(want, want), "got": got,
                                   "got_name": names.get(got, got), "lang": lang, "count": 0, "examples": []})
        gp["count"] += 1
        if len(gp["examples"]) < 4:
            gp["examples"].append(t)
    out = sorted(groups.values(), key=lambda x: -x["count"])[:limit]
    return {"groups": out, "total": miss, "checked": len(rows),
            "accuracy": round(100 * (1 - miss / max(1, len(rows))), 1)}


def balance_plan() -> list:
    """How many labels each agent should have to match its real traffic — the target to teach toward."""
    labels = inbox._labels()
    n = len(labels)
    tr = _blind_share()
    by_agent: dict = {}
    for l in labels:
        by_agent[l.get("agent") or l.get("intent") or "?"] = by_agent.get(l.get("agent") or l.get("intent") or "?", 0) + 1
    g = router.graph()
    names = {x["id"]: x["name"] for x in g["nodes"] if x["type"] == "agent"}
    plan = []
    total = max(n, 20)
    for a, share in sorted(tr.items(), key=lambda kv: -kv[1]):
        have = by_agent.get(a, 0)
        want = round(share * total)
        plan.append({"agent": a, "agent_name": names.get(a, a), "have": have, "target": want,
                     "add": max(0, want - have)})
    return plan


def add_example(text: str, agent: str, by: str) -> dict:
    """Add one training example (a label), refusing anything that copies the blind exam — so accuracy stays honest."""
    t = " ".join((text or "").split())
    if len(t) < 3:
        raise ValueError("write a full sentence")
    if inbox.key(t) in _blind_keys():
        raise ValueError("that sentence is in the blind exam — teach a different one of the same kind, or the score would be dishonest")
    return inbox.label_text(t, agent, by, source="doctor")


def generate(agent: str, n: int = 6, hint: str = "") -> dict:
    """Claude drafts fresh example sentences for an agent (grounded in its purpose + examples), for a human to
    accept. Never auto-added, and blind-exam sentences are filtered out."""
    import llm
    g = router.graph()
    node = router.node(g, agent)
    if not node:
        raise ValueError("no such agent")
    examples = [e for e in (node.get("examples") or []) if isinstance(e, str)][:10]
    kind = node.get("kind")
    sys = ("You write short, realistic customer messages for a payments app, to train a router. Return ONLY the "
           "messages, one per line, no numbering. Vary wording, length and (where natural) language "
           "(English, Hinglish, Hindi, Arabic). Each must clearly belong to the given category.")
    prompt = (f"Agent: {node['name']} (kind: {kind}).\n"
              f"{'Existing examples:' + chr(10) + chr(10).join('- ' + e for e in examples) if examples else ''}\n"
              f"{'Focus: ' + hint if hint else ''}\nWrite {max(3, min(12, n))} new messages a customer would send that should go to this agent.")
    raw = llm.complete(sys, prompt, max_tokens=700)
    blind = _blind_keys()
    seen, out = set(), []
    for line in raw.splitlines():
        s = re.sub(r"^[\s\-*•\d.)]+", "", line).strip().strip('"')
        k = inbox.key(s)
        if len(s) >= 3 and k not in seen and k not in blind:
            seen.add(k)
            out.append(s)
    return {"agent": agent, "agent_name": node["name"], "suggestions": out[:n]}


def _blind_sig(v) -> tuple:
    """A fingerprint of which blind set a head was scored on (its per-agent denominators), so we never compare
    heads measured on different blind sets — that was making an old head look like the champion."""
    bpa = (v.get("scores") or {}).get("blind_per_agent") or {}
    return tuple(sorted((a, str(s).split("/")[-1]) for a, s in bpa.items()))


def champion() -> dict:
    """Is the live head still the best we have trained — among heads scored on the SAME blind set?
    (Stops people chasing retrains that lose accuracy, and stops false 'promote the old one' advice.)"""
    rl = store.versions()["router"]
    act = rl.get("active")
    heads = [v for v in rl.get("list", []) if (v.get("scores") or {}).get("blind") is not None]
    live = next((v for v in heads if v["id"] == act), None)
    if not live:
        return {"active": act, "best": None, "live_is_best": True, "newer_worse": []}
    sig = _blind_sig(live)
    cohort = [v for v in heads if _blind_sig(v) == sig]      # only heads on the live head's blind set
    best = max(cohort, key=lambda v: v["scores"]["blind"])
    worse = [{"id": v["id"], "blind": v["scores"]["blind"]} for v in cohort
             if v["id"] != act and v["scores"]["blind"] < live["scores"]["blind"]]
    return {"active": act, "active_blind": live["scores"]["blind"], "best": best["id"], "best_blind": best["scores"]["blind"],
            "live_is_best": best["id"] == act, "newer_worse": worse, "cohort": [v["id"] for v in cohort]}
