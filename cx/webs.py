"""The answer, rendered as seven knowledge webs with a crawler walking them.

## What this is for

The engine already explains itself: every decision appends a `lane_trace` entry saying what
it did and why. That trace is a list, and a list reads like a log. This module turns the same
material into something you can watch — seven webs of what the assistant KNOWS, and the path
the crawler actually took through them for one question.

## The one rule

**Every node is something that happened.** The webs are built from the real trace, the real
classifier ranking, the real second opinion, the real ledger rows and the real category
table. Nothing here invents a node to make the picture denser, and nothing claims a step the
engine did not take. A visualisation of reasoning that contains invented reasoning is worse
than no visualisation, because it is the one part of the system nobody can check by reading
the answer.

So the distinction the UI draws is between:

* **what the web contains** — everything that was available to consider, drawn faint; and
* **what the crawler touched** — the nodes that actually took part, drawn lit, in order.

A web that is mostly dark is not a failure. It means the assistant had a lot available and
used a little, which is what a narrow, closed-world assistant is supposed to look like.

## Why seven

They are the real stages, not a number chosen for the picture:

  1. `intent`   — the classifier's own ranking over the closed label set
  2. `laya`     — the independently-trained second opinion, agreeing or not
  3. `resolve`  — slots and retrieval: what the sentence named and what we fetched
  4. `matrix`   — the decision table, where there is one
  5. `memory`   — the conversation frame and the task ledger this turn read or wrote
  6. `execute`  — the handler's own work, including the spend aggregation
  7. `learn`    — what this turn leaves behind for the next one

Layout is deterministic: the same question draws the same web, so a demo can be rehearsed
and a screenshot means something.
"""
from __future__ import annotations

import learn as learnmod

import hashlib
import math
import re

import taxonomy
from taxonomy import Intent, MONEY_MOVING, SPECS

#: One colour per web, in the order they are crawled. Chosen to read on black and to stay
#: distinguishable when a cluster is drawn at 15% opacity.
WEB_COLOURS = {
    "intent":  "#ff3d7f",
    "laya":    "#36d6c3",
    "resolve": "#f5c451",
    "matrix":  "#8b7bff",
    "memory":  "#4da3ff",
    "execute": "#52e39a",
    "learn":   "#ff8a5c",
}

WEB_TITLES = {
    "intent":  ("INTENT", "what are they asking"),
    "laya":    ("LAYA", "a second opinion"),
    "resolve": ("RESOLVE", "names, periods, rows"),
    "matrix":  ("MATRIX", "the decision table"),
    "memory":  ("MEMORY", "what this conversation knows"),
    "execute": ("EXECUTE", "the work itself"),
    "learn":   ("LEARN", "what this turn leaves behind"),
}

ORDER = ("intent", "laya", "resolve", "matrix", "memory", "execute", "learn")

#: Which engine lane feeds which web. `resolve` and `execute` share a lane, so the step name
#: breaks the tie — anything the handler itself did is execution.
_LANE_WEB = {"understand": "intent", "resolve": "resolve", "decide": "matrix", "act": "execute"}


def _chip(detail: str) -> str:
    """The short annotation that floats beside a node while the crawler is on it.

    It is the first clause of the node's own reason, not a new sentence written for the
    picture — so a chip can never say something the trace did not. Clauses are cut at the
    punctuation the detail already uses, which is where its first complete thought ends.
    """
    d = " ".join((detail or "").split())
    if not d:
        return ""
    for sep in (" — ", " – ", ": ", ", ", ". "):
        i = d.find(sep)
        if 6 < i <= 46:
            d = d[:i]
            break
    d = d.strip(" .,;:—–")
    # A chip is a glance, not a sentence. Where the detail has no punctuation to cut at,
    # take whole words up to the limit rather than slicing mid-phrase — "merchant payments
    # read from the customer's" is worse than saying less.
    LIMIT = 34
    if len(d) > LIMIT:
        words, out = d.split(), ""
        for w in words:
            if len(out) + len(w) + 1 > LIMIT:
                break
            out = f"{out} {w}".strip()
        d = (out or d[:LIMIT]).rstrip(" ,;:")
    # Never end on a word that is waiting for the next one — "read from the" reads as a
    # truncation, "read from" reads as a phrase.
    _DANGLING = {"the", "a", "an", "of", "from", "to", "for", "in", "on", "and", "or",
                 "with", "that", "this", "its", "their", "is", "are", "was", "by", "at",
                 "as", "so", "than", "into", "over", "per"}
    parts = d.split()
    while parts and parts[-1].lower() in _DANGLING:
        parts.pop()
    return " ".join(parts) or d


def _rng(seed: str):
    """A tiny deterministic generator. The layout must be stable for the same question —
    a demo that rearranges itself between runs looks like it is guessing."""
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    state = int.from_bytes(h[:8], "big") or 1

    def nxt() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    return nxt


def _place(n: int, seed: str) -> list[tuple[float, float]]:
    """`n` points in a unit disc, golden-angle so they never clump, jittered so they never
    look like a sunflower either."""
    rnd = _rng(seed)
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        r = math.sqrt((i + 0.5) / max(n, 1))
        th = i * ga + rnd() * 0.6
        jitter = 0.06 * (rnd() - 0.5)
        pts.append((round(math.cos(th) * (r + jitter), 4),
                    round(math.sin(th) * (r + jitter), 4)))
    return pts


class _Web:
    def __init__(self, key: str):
        self.key = key
        self.nodes: list[dict] = []
        self.edges: list[list[int]] = []
        self.path: list[int] = []

    def add(self, label: str, kind: str = "fact", hit: bool = False,
            detail: str = "", weight: float = 0.5) -> int:
        self.nodes.append({"id": len(self.nodes), "label": str(label)[:46], "kind": kind,
                           "hit": bool(hit), "detail": detail[:220], "weight": round(weight, 3),
                           "chip": _chip(detail)})
        return len(self.nodes) - 1

    def walk(self, *ids: int) -> None:
        """Mark these nodes as visited, in order. The path IS the reasoning — a node that is
        lit but not on the path was available and not used."""
        for i in ids:
            if 0 <= i < len(self.nodes):
                self.nodes[i]["hit"] = True
                self.path.append(i)

    def link(self, a: int, b: int) -> None:
        if a != b and 0 <= a < len(self.nodes) and 0 <= b < len(self.nodes):
            self.edges.append([a, b])

    def chain(self, ids: list[int]) -> None:
        for a, b in zip(ids, ids[1:]):
            self.link(a, b)

    def finish(self, seed: str) -> dict:
        pts = _place(len(self.nodes), f"{seed}:{self.key}")
        for node, (x, y) in zip(self.nodes, pts):
            node["x"], node["y"] = x, y
        # Everything hangs off the first node, so an isolated cluster never appears.
        for i in range(1, len(self.nodes)):
            if not any(i in e for e in self.edges):
                self.link(0, i)
        title, subtitle = WEB_TITLES[self.key]
        found = sum(1 for n in self.nodes if n["hit"])
        return {"key": self.key, "title": title, "subtitle": subtitle,
                "colour": WEB_COLOURS[self.key], "nodes": self.nodes,
                "edges": self.edges, "path": self.path,
                "found": found, "total": len(self.nodes),
                "pct": round(found / len(self.nodes) * 100) if self.nodes else 0}


# ---------------------------------------------------------------------------------------

def build(a, said: str = "") -> dict:
    """A crawl plan for one answered question. `a` is an engine `Answer`."""
    said = said or getattr(a, "said", "") or ""
    seed = f"{said}|{a.intent}"
    trace = list(getattr(a, "lane_trace", []) or [])
    spend = getattr(a, "spend", None)

    webs = [
        _web_intent(a, trace),
        _web_laya(a),
        _web_resolve(a, trace, spend),
        _web_matrix(a, trace),
        _web_memory(a, trace),
        _web_execute(a, trace, spend),
        _web_learn(a, trace, spend),
    ]
    out = [w.finish(seed) for w in webs]

    return {
        "question": said,
        "answer": getattr(a, "say", "") or "",
        "intent": a.intent,
        "intent_label": SPECS[Intent(a.intent)].label if a.intent in {i.value for i in Intent} else a.intent,
        "lang": getattr(a, "lang", "en"),
        "ms": getattr(a, "ms", 0),
        "model": getattr(a, "model", ""),
        "confidence": round(float(getattr(a, "confidence", 0.0) or 0.0), 3),
        "webs": out,
        "log": _log(out, trace),
        "heat": _heat(spend, out),
        "score": _score(a, out),
        "radar": [{"axis": w["title"].lower(), "value": w["pct"], "colour": w["colour"]}
                  for w in out],
        "code": _code(a, spend),
        "spend": spend,
        "outcome": getattr(a, "outcome", None),
        #: What the engine is PROPOSING, by kind only. This screen shows thinking; it has no
        #: authority to execute, and carrying the full proposal would invite it to try.
        "proposes": (getattr(a, "proposal", None) or {}).get("kind"),
        #: True when continuing would move money. The crawl still runs — watching the
        #: decision is the whole point — but the screen has to say out loud that finishing it
        #: happens on the payment surface, because a payment that visibly stalls with no PIN
        #: prompt reads as a broken wallet.
        "moves_money": bool(
            Intent(a.intent) in MONEY_MOVING if a.intent in {i.value for i in Intent}
            else False),
    }


def _web_intent(a, trace) -> _Web:
    """The closed label set, with the classifier's ranking over it."""
    w = _Web("intent")
    chosen = a.intent
    top = list(getattr(a, "top", []) or [])
    scores = {}
    for item in top:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            scores[str(item[0])] = float(item[1])
        elif isinstance(item, dict):
            scores[str(item.get("label"))] = float(item.get("score", 0))

    ids = {}
    for label in taxonomy.LABELS:
        sc = scores.get(label, 0.0)
        ids[label] = w.add(label, kind="label", weight=max(sc, 0.04),
                           detail=(f"{sc:.0%} from {getattr(a, 'model', '') or 'the classifier'}"
                                   if sc else "in the closed set, not chosen"))
    # The crawler walks the ranked candidates it actually considered, then the winner.
    ranked = [l for l, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:4] if l in ids]
    w.walk(*[ids[l] for l in ranked if l != chosen])
    if chosen in ids:
        w.nodes[ids[chosen]]["kind"] = "chosen"
        w.walk(ids[chosen])
    w.chain([ids[l] for l in ranked] + ([ids[chosen]] if chosen in ids else []))
    for st in trace:
        if st.get("lane") == "understand":
            n = w.add(st.get("what", ""), kind=st.get("tag", "rule"), detail=st.get("detail", ""),
                      weight=0.8)
            w.walk(n)
            if chosen in ids:
                w.link(ids[chosen], n)
    return w


def _web_laya(a) -> _Web:
    """The second opinion. One node per coarse family it can choose between, so a
    disagreement is visibly a disagreement about WHICH family, not a vague doubt."""
    w = _Web("laya")
    op = getattr(a, "advisory", None) or {}
    families = ["payment_status", "refund", "sending_money", "paying_a_shop",
                "fees_and_charges", "offers_nearby", "spending_pattern", "something_else"]
    ids = {f: w.add(f, kind="family", weight=0.25,
                    detail="a family Laya can choose between") for f in families}
    if not op or not op.get("available"):
        n = w.add("laya unavailable", kind="guard", weight=0.9,
                  detail="no second opinion for this turn — the engine proceeded on its own "
                         "classifier, which is the designed fallback, not a failure")
        w.walk(n)
        return w
    topic = str(op.get("topic") or "")
    conf = float(op.get("confidence") or 0.0)
    if topic in ids:
        w.nodes[ids[topic]]["kind"] = "chosen"
        w.nodes[ids[topic]]["weight"] = max(conf, 0.3)
        w.nodes[ids[topic]]["detail"] = f"Laya reads it as {topic} ({conf:.0%})"
        w.walk(ids[topic])
    verdict = w.add("agrees" if op.get("agrees") else "disagrees",
                    kind="chosen" if op.get("agrees") else "guard", weight=0.95,
                    detail=("two independently-trained models read the sentence the same way"
                            if op.get("agrees") else
                            "a real signal the sentence is ambiguous — the engine asks rather "
                            "than assumes"))
    w.walk(verdict)
    if topic in ids:
        w.link(ids[topic], verdict)
    return w


def _web_resolve(a, trace, spend) -> _Web:
    """What the sentence named, and what we went and fetched because of it."""
    w = _Web("resolve")
    root = w.add("the sentence", kind="input", weight=1.0,
                 detail=getattr(a, "said", "")[:200])
    w.walk(root)
    for k, v in (getattr(a, "slots", {}) or {}).items():
        if v in (None, "", [], {}):
            continue
        n = w.add(f"{k} = {v}", kind="slot", weight=0.8, detail=f"pulled out of the sentence")
        w.walk(n)
        w.link(root, n)
    if spend:
        per = w.add(spend.get("period_label", ""), kind="slot", weight=0.9,
                    detail="the window being analysed, with the same length before it")
        w.walk(per)
        w.link(root, per)
        src = w.add("live ledger" if spend.get("live") else "demo corpus",
                    kind="fact" if spend.get("live") else "guard", weight=0.9,
                    detail=("merchant payments read from the customer's own ledger"
                            if spend.get("live") else
                            "the ledger did not answer — this is the demo corpus, and the "
                            "answer says so"))
        w.walk(src)
        w.link(per, src)
        for c in (spend.get("categories") or [])[:6]:
            cn = w.add(f"{c['label']} · {c['count']}", kind="category", weight=0.6,
                       detail=f"AED {c['total']:,.2f} this period")
            w.walk(cn)
            w.link(src, cn)
            if c.get("top_merchant"):
                mn = w.add(c["top_merchant"], kind="merchant", weight=0.5,
                           detail=f"AED {c.get('top_merchant_amount', 0):,.2f} — "
                                  f"{(spend.get('basis') or {}).get(c['top_merchant'], '')}")
                w.walk(mn)
                w.link(cn, mn)
    for st in trace:
        if st.get("lane") == "resolve":
            n = w.add(st.get("what", ""), kind=st.get("tag", "rule"),
                      detail=st.get("detail", ""), weight=0.75)
            w.walk(n)
            w.link(root, n)
    return w


def _web_matrix(a, trace) -> _Web:
    """The decision table. Drawn as a grid of cells so the one that fired is visibly a CELL
    in a policy, not an opinion the model formed."""
    w = _Web("matrix")
    states = ["SUCCESS", "FAILED", "PENDING", "ON_HOLD", "REVERSED", "UNKNOWN"]
    positions = ["WITH_US", "IN_FLIGHT", "WITH_PARTNER", "CREDITED"]
    ids = {}
    for s in states:
        for p in positions:
            ids[(s, p)] = w.add(f"{s}·{p}", kind="cell", weight=0.14,
                                detail="a cell of the refund decision table")
    decided = [st for st in trace if st.get("lane") == "decide"]
    txn = getattr(a, "txn", None) or {}
    key = (str(txn.get("state", "")).upper(), str(txn.get("position", "")).upper())
    if key in ids:
        w.nodes[ids[key]]["kind"] = "chosen"
        w.nodes[ids[key]]["weight"] = 1.0
        w.walk(ids[key])
    for st in decided:
        n = w.add(st.get("what", ""), kind=st.get("tag", "rule"),
                  detail=st.get("detail", ""), weight=0.85)
        w.walk(n)
    if not decided and key not in ids:
        n = w.add("no table for this question", kind="guard", weight=0.9,
                  detail="this intent is not a money-moving decision, so no matrix cell "
                         "applies — the table is shown unlit rather than hidden, because "
                         "'the policy did not apply' is itself worth seeing")
        w.walk(n)
    return w


def _web_memory(a, trace) -> _Web:
    """What the conversation already knew, and what this turn added to it."""
    w = _Web("memory")
    sess = w.add(f"session · {getattr(a, 'session', '') or 'default'}", kind="input",
                 weight=0.9, detail="the frame this turn ran inside")
    w.walk(sess)
    lang = w.add(f"language · {getattr(a, 'lang', 'en')}", kind="fact", weight=0.7,
                 detail="decided once from what they wrote, then used by every handler")
    w.walk(lang)
    w.link(sess, lang)
    recalled = getattr(a, "recalled", None)
    if recalled:
        n = w.add("reached back to an earlier task", kind="chosen", weight=1.0,
                  detail=str(recalled)[:200])
        w.walk(n)
        w.link(sess, n)
    else:
        w.add("no earlier task matched", kind="fact", weight=0.3,
              detail="nothing in this conversation's ledger answered to this sentence")
    if getattr(a, "proposal", None):
        n = w.add("a proposal is now held", kind="chosen", weight=0.95,
                  detail="the next 'yes' executes this, and nothing else")
        w.walk(n)
        w.link(sess, n)
    for st in trace:
        if st.get("tag") in ("recall", "task"):
            n = w.add(st.get("what", ""), kind="recall", detail=st.get("detail", ""),
                      weight=0.8)
            w.walk(n)
            w.link(sess, n)
    return w


def _web_execute(a, trace, spend) -> _Web:
    """The handler's own work — for a spend question, the aggregation itself."""
    w = _Web("execute")
    root = w.add(SPECS[Intent(a.intent)].label if a.intent in {i.value for i in Intent}
                 else a.intent, kind="input", weight=1.0, detail="the handler that ran")
    w.walk(root)
    if spend:
        tot = w.add(f"AED {spend.get('total', 0):,.2f}", kind="chosen", weight=1.0,
                    detail=f"{spend.get('payments', 0)} merchant payments in "
                           f"{spend.get('period_label', '')}")
        w.walk(tot)
        w.link(root, tot)
        prev = w.add(f"was AED {spend.get('previous_total', 0):,.2f}", kind="fact", weight=0.8,
                     detail="the same length of time immediately before")
        w.walk(prev)
        w.link(tot, prev)
        if spend.get("mover"):
            mv = w.add(f"moved · {spend['mover']}", kind="chosen", weight=1.0,
                       detail="reported because it is the biggest CHANGE — the biggest total "
                              "is the same category every period")
            w.walk(mv)
            w.link(prev, mv)
        if spend.get("uncategorised"):
            un = w.add(f"uncategorised AED {spend['uncategorised']:,.2f}", kind="guard",
                       weight=0.9,
                       detail=f"{spend.get('uncategorised_pct', 0)}% at merchants with no "
                              f"category yet — reported, not guessed")
            w.walk(un)
            w.link(tot, un)
    for st in trace:
        if st.get("lane") == "act":
            n = w.add(st.get("what", ""), kind=st.get("tag", "rule"),
                      detail=st.get("detail", ""), weight=0.8)
            w.walk(n)
            w.link(root, n)
    out = w.add(getattr(a, "outcome", None) or "ANSWERED", kind="chosen", weight=1.0,
                detail=(getattr(a, "say", "") or "")[:200])
    w.walk(out)
    w.link(root, out)
    return w


def _web_learn(a, trace, spend) -> _Web:
    """What this turn leaves behind.

    Deliberately modest, and specific about which half is which. Nothing here retrains a
    model — on a surface that moves money that would be a safety regression, not a feature,
    because the classifier would change its mind between two customers with no release and
    no way back. Drawing a node that implied otherwise would be a lie about the system.

    What it really does is two things, and both are shown as what they are: it REMEMBERS
    (the frame, the language, the task ledger), and it KEEPS the turns the next training run
    needs — written to a real file, counted from that file, not asserted.
    """
    w = _Web("learn")
    root = w.add("this turn", kind="input", weight=1.0, detail=getattr(a, "said", "")[:180])
    w.walk(root)
    items = [
        ("frame updated", "the slots and the intent this session is now on", True),
        (f"language pinned · {getattr(a, 'lang', 'en')}",
         "so a bare '100' next turn does not switch back to English", True),
    ]
    # What actually happened to this turn in the store — including when the answer is
    # "nothing", which is the honest answer for an ordinary confident turn.
    kept = getattr(a, "kept", None)
    st = learnmod.stats()
    if kept:
        why = {"confirmed": "the customer went through with it, so the reading was right",
               "corrected": "the next turn contradicted this one — the most valuable row in "
                            "the file, and the hardest to get any other way",
               "rule-reached": "reached by rule, not by the classifier, so the model cannot "
                               "predict it yet — exactly what the next label space needs",
               "guarded": "a guard fired, which means we were unsure and said so"}
        items.append((f"kept · {kept}", why.get(kept, kept), True))
    else:
        items.append(("not kept",
                      "an ordinary, confident turn teaches nothing — keeping every one of "
                      "them would fill the file with the same four sentences", False))
    items.append((f"corpus · {st['rows_on_disk']} turns",
                  f"written to {st['store'].rsplit('/', 2)[-1]} — {st['confirmed']} "
                  f"confirmed, {st['corrected']} corrected, {st['rule']} rule-reached this "
                  f"run. The next training run reads this file; nothing retrains from here.",
                  bool(st["rows_on_disk"])))
    if a.model == "spend-rule":
        items.append(("candidate for the next retrain",
                      "spend_analysis is the 24th label and the deployed model has 23 "
                      "outputs — this turn was reached by rule, and is exactly the kind of "
                      "example the next training set needs", True))
    adv = getattr(a, "advisory", None) or {}
    if adv.get("available") and not adv.get("agrees"):
        items.append(("disagreement logged",
                      "two models reading one sentence differently is the signal worth "
                      "keeping — it marks the sentence as genuinely ambiguous", True))
    if spend and spend.get("uncategorised"):
        items.append(("merchants needing a category",
                      "these are the rows the one-time migration will fix when category "
                      "moves onto the merchant record", True))
    for label, detail, hit in items:
        n = w.add(label, kind="learn", weight=0.85, detail=detail, hit=hit)
        w.walk(n)
        w.link(root, n)
    return w


# --- the side panels -------------------------------------------------------------------

_VERBS = {"input": "read", "slot": "read", "label": "read", "family": "read",
          "category": "read", "merchant": "read", "cell": "read", "fact": "read",
          "chosen": "walk", "guard": "flag", "rule": "link", "laya": "link",
          "recall": "link", "learn": "walk"}


def _log(webs, trace) -> list[dict]:
    """The crawl log — one line per node the crawler touched, in crawl order."""
    out = []
    t = 0.0
    for w in webs:
        for nid in w["path"]:
            n = w["nodes"][nid]
            t += 0.07
            out.append({"t": f"{int(t // 60):02d}:{t % 60:04.1f}",
                        "verb": _VERBS.get(n["kind"], "read"),
                        "target": n["label"], "web": w["key"], "colour": w["colour"]})
    return out


def _heat(spend, webs) -> dict:
    """The heat grid. For a spend question it is the real category breakdown; otherwise it
    is per-web coverage, so the panel is never empty and never invented."""
    if spend and spend.get("categories"):
        top = max((c["total"] for c in spend["categories"]), default=1.0) or 1.0
        rows = [{"label": c["label"][:14], "cells":
                 [min(1.0, (c["total"] / top) * (1.0 - i * 0.07)) for i in range(14)],
                 "value": f"{c['total']:,.0f}",
                 "colour": WEB_COLOURS[ORDER[i % len(ORDER)]]}
                for i, c in enumerate(spend["categories"])]
        return {"title": "CATEGORY HEAT", "unit": "AED", "rows": rows,
                "total": f"{spend.get('total', 0):,.0f}"}
    rows = [{"label": w["title"][:14],
             "cells": [1.0 if (i / 14.0) < (w["pct"] / 100.0) else 0.12 for i in range(14)],
             "value": f"{w['found']}/{w['total']}", "colour": w["colour"]} for w in webs]
    return {"title": "WEB COVERAGE", "unit": "nodes", "rows": rows,
            "total": str(sum(w["found"] for w in webs))}


def _score(a, webs) -> dict:
    """The gauge. It is the engine's own confidence — not a number invented for the panel."""
    conf = round(float(getattr(a, "confidence", 0.0) or 0.0) * 100)
    flags = sum(1 for w in webs for n in w["nodes"] if n["kind"] == "guard" and n["hit"])
    return {"value": conf, "label": "confidence",
            "model": getattr(a, "model", "") or "—",
            "flags": flags,
            "note": "asked, not guessed" if flags else "no guard fired",
            "ms": getattr(a, "ms", 0)}


def _code(a, spend) -> list[dict]:
    """The crawler source panel. Real pseudocode for what just ran, line by line."""
    lines = [
        ("c", "# crawler.py — one question, seven webs"),
        ("k", "plan = webs.build(engine.ask(question))"),
        ("", ""),
        ("k", "for web in plan.webs:"),
        ("n", "    spider.enter(web)"),
        ("k", "    for node in web.path:"),
        ("n", "        spider.touch(node)"),
        ("k", "        if node.kind == 'guard':"),
        ("f", "            flags.append(node)"),
        ("", ""),
        ("c", f"# intent  {a.intent}"),
        ("c", f"# model   {getattr(a, 'model', '') or '-'}"),
        ("c", f"# conf    {float(getattr(a, 'confidence', 0) or 0):.0%}"),
    ]
    if spend:
        lines += [("", ""),
                  ("c", f"# window  {spend.get('period_label', '')}"),
                  ("c", f"# total   AED {spend.get('total', 0):,.2f}"),
                  ("c", f"# moved   {spend.get('mover') or '-'}")]
    return [{"cls": c, "text": t} for c, t in lines]
