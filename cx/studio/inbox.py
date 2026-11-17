"""The "Teach me" inbox: only the decisions worth a person's time, and the rules that keep the models
from ever being trained wrong.

What reaches the inbox (from the ledger — console runs and shadowed app traffic):
  thumbs_down      someone said the answer was wrong
  shadow_mismatch  the Orchestrator would have sent a real app message somewhere other than the engine
  disagree         the reflex and the reasoning model picked different agents
  intent_unsure    the intent model was under 75% sure about a real app message
  unsure / handoff the Orchestrator itself was unsure, or a knowledge agent handed the message on
Near-duplicates are grouped (they count once, with their size) and ranked by reason × frequency.

What may TRAIN (training_rows) — the safety rules:
  1. only HUMAN labels (an admin's click); a model's own guess never becomes training data
  2. never a message the guardrails stopped (scams, injection, credential fishing), never a fragment
     with no real words, never anything marked "not a real question"
  3. a sentence labelled differently by two people is frozen until someone resolves it
  4. an intent label must be one of the taxonomy's intents; agent labels must be existing agents
  (5. the training job caps how much a run can be shifted by new labels — train_job.py)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import ledger
import router
import store

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = "labels.jsonl"
DISMISSED = "inbox_dismissed.json"
WEIGHT = {"redteam_breach": 5, "unsupported": 4, "thumbs_down": 5, "shadow_mismatch": 4, "intent_unsure": 3, "disagree": 3, "handoff": 2, "unsure": 2, "new_route": 1}
WHY = {"redteam_breach": "the red team got through — the live route did the wrong thing", "unsupported": "the answer said something its documents do not", "thumbs_down": "someone marked the answer wrong", "shadow_mismatch": "the Orchestrator and the live engine disagree",
       "intent_unsure": "the intent model was unsure", "disagree": "the reflex and the reasoning model disagree",
       "handoff": "a knowledge agent handed it on", "unsure": "the Orchestrator was unsure",
       "new_route": "the Orchestrator would use a knowledge agent here"}


def key(text: str) -> str:
    t = re.sub(r"[^\w\sऀ-ॿ؀-ۿ]", " ", (text or "").lower())
    return " ".join(t.split())[:200]


def intent_labels() -> list[str]:
    try:
        from taxonomy import LABELS as L
        return list(L)
    except Exception:                                       # noqa: BLE001
        return []


def _labels() -> list[dict]:
    p = os.path.join(store.DATA, LABELS)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def _dismissed() -> dict:
    return store.read(DISMISSED, {}) or {}


def _reasons(r: dict, tr: dict, g: dict) -> list[str]:
    out = []
    if r.get("source") == "redteam":                        # only a confirmed breach is worth a person's label
        return ["redteam_breach"] if (tr.get("redteam") or {}).get("breach") else []
    if r.get("feedback") == -1:
        out.append("thumbs_down")
    eng = tr.get("engine") or {}
    if r["source"] == "shadow" and r.get("engine_agent") and r["engine_agent"] != "guardrails" and r.get("chosen"):
        n = router.node(g, r["chosen"]) or {}
        if r["chosen"] != r["engine_agent"]:
            out.append("new_route" if n.get("kind") == "knowledge" and r["engine_agent"] == "action" else "shadow_mismatch")
    if eng.get("confidence") is not None and eng.get("confidence") < 0.75 and r["source"] == "shadow" and eng.get("intent") != "out_of_scope":
        out.append("intent_unsure")
    rx, ly = tr.get("reflex") or {}, tr.get("laya") or {}
    if rx.get("available") and ly.get("available") and rx.get("choice") and ly.get("choice") and rx["choice"] != ly["choice"]:
        out.append("disagree")
    if tr.get("handoffs"):
        out.append("handoff")
    fs = r.get("faith_score")
    if fs is not None and fs < 0.67:
        out.append("unsupported")
    if r.get("by") in ("reflex (unsure)", "fallback"):
        out.append("unsure")
    return out


def items(limit: int = 60) -> dict:
    g = router.graph()
    done = {key(l["text"]) for l in _labels()} | set(_dismissed())
    groups: dict[str, dict] = {}
    for r in ledger.recent(500):
        if r.get("source") == "sandbox":                    # Academy practice never becomes a lesson
            continue
        if r.get("label") or not r.get("text") or r.get("by") == "guardrails" and r.get("chosen") is None:
            continue
        k = key(r["text"])
        if not k or k in done:
            continue
        full = ledger.get(r["id"])
        tr = full["trace"]
        rs = _reasons(full, tr, g)
        if not rs:
            continue
        it = groups.get(k)
        if not it:
            eng = tr.get("engine") or {}
            it = groups[k] = {"key": k, "text": r["text"], "ids": [], "lang": r.get("lang"), "reasons": set(), "count": 0, "last": 0,
                              "chosen": r.get("chosen"), "by": r.get("by"), "confidence": r.get("confidence"),
                              "engine_agent": r.get("engine_agent"), "engine_intent": eng.get("intent") or r.get("engine_intent"),
                              "engine_confidence": eng.get("confidence"), "source": r["source"],
                              "laya": (tr.get("laya") or {}).get("choice"), "reflex": (tr.get("reflex") or {}).get("choice")}
        it["ids"].append(r["id"])
        it["count"] += 1
        it["last"] = max(it["last"], r["at"])
        it["reasons"] |= set(rs)
    out = []
    for it in groups.values():
        it["reasons"] = sorted(it["reasons"], key=lambda x: -WEIGHT[x])
        it["why"] = [WHY[x] for x in it["reasons"]]
        it["score"] = round(sum(WEIGHT[x] for x in it["reasons"]) * (1 + math.log(it["count"])), 2)
        out.append(it)
    out.sort(key=lambda x: (-x["score"], -x["last"]))
    conflicts = conflicts_list()
    return {"items": out[:limit], "total": len(out), "conflicts": conflicts, "labelled": len(_labels()),
            "agents": {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}, "intents": intent_labels()}


def label(text_key: str, ids: list[int], agent: str | None, intent: str | None, by: str, golden: bool = False) -> dict:
    g = router.graph()
    if agent and not router.node(g, agent):
        raise ValueError("no such agent")
    if intent and intent not in intent_labels():
        raise ValueError("not one of the intent model's intents")
    if not agent and not intent:
        raise ValueError("pick the right agent (and, for the action agent, the intent)")
    rows = [ledger.get(int(i)) for i in ids[:200]]
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("those decisions are gone (older than 30 days?)")
    text = rows[0]["text"]
    for r in rows:
        ledger.set_label(r["id"], agent or (r.get("chosen") or ""), by)
    rec = {"text": text, "key": key(text), "agent": agent, "intent": intent, "by": by, "at": time.time(),
           "ids": [r["id"] for r in rows], "lang": router.language(text), "source": rows[0]["source"]}
    with open(store.path(LABELS), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if golden and agent:
        import gates
        gates.add(text, agent, by, note="from the inbox", source="inbox")
    return rec


def label_text(text: str, agent: str, by: str, source: str = "arena", golden: bool = False) -> dict:
    """A label for a sentence that is not a ledger decision (the boundary arena) — same rules apply."""
    if not router.node(router.graph(), agent):
        raise ValueError("no such agent")
    text = " ".join((text or "").split())[:400]
    rec = {"text": text, "key": key(text), "agent": agent, "intent": None, "by": by, "at": time.time(), "ids": [],
           "lang": router.language(text), "source": source}
    with open(store.path(LABELS), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if golden:
        import gates
        gates.add(text, agent, by, note=f"from the {source}", source=source)
    return rec


def dismiss(text_key: str, why: str, by: str) -> None:
    d = _dismissed()
    d[text_key] = {"why": why[:80], "by": by, "at": time.time()}
    store.write(DISMISSED, d)


def remove(text_key: str, by: str = "") -> int:
    """Drop every label for a sentence from the training set (the Label Doctor). The removed rows are kept in a
    timestamped backup beside labels.jsonl, so a mistaken drop can be restored by hand."""
    p = store.path(LABELS)
    if not os.path.exists(p):
        return 0
    rows = _labels()
    keep = [l for l in rows if l.get("key") != text_key]
    dropped = len(rows) - len(keep)
    if not dropped:
        return 0
    with open(p + ".bak-" + time.strftime("%Y%m%d%H%M%S"), "w", encoding="utf-8") as fh:
        for l in rows:
            fh.write(json.dumps(l, ensure_ascii=False) + "\n")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for l in keep:
            fh.write(json.dumps(l, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return dropped


def conflicts_list() -> list[dict]:
    by_key: dict[str, list[dict]] = {}
    for l in _labels():
        by_key.setdefault(l["key"], []).append(l)
    out = []
    for k, ls in by_key.items():
        agents = {l["agent"] for l in ls if l.get("agent")}
        intents = {l["intent"] for l in ls if l.get("intent")}
        if len(agents) > 1 or len(intents) > 1:
            out.append({"key": k, "text": ls[-1]["text"], "agents": sorted(agents), "intents": sorted(intents), "by": sorted({l["by"] for l in ls})})
    return out


_WORD = re.compile(r"[^\W\d_]{3,}", re.U)


def training_rows(system: str, until: float | None = None) -> list[dict]:
    """Human labels that are safe to train on (the rules at the top of this file). until: as of that time."""
    frozen = {c["key"] for c in conflicts_list()}
    dismissed = set(_dismissed())
    out, seen = [], set()
    for l in reversed([x for x in _labels() if until is None or (x.get("at") or 0) <= until]):    # newest label per sentence wins
        k = l["key"]
        if k in seen or k in frozen or k in dismissed:
            continue
        seen.add(k)
        if not _WORD.search(l["text"]):                     # a fragment with no real words
            continue
        if system == "orchestrator" and l.get("agent"):
            out.append(l)
        if system == "intent" and l.get("intent"):
            out.append(l)
    return out


def new_labels_since(system: str, ts: float) -> int:
    return sum(1 for l in training_rows(system) if l["at"] > ts)
