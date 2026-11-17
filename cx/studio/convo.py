"""Conversation Multiverse: whole conversations replayed through every router version, turn by turn.

The single-request Multiverse asks "would this sentence go somewhere else?". A conversation is harder: every turn is
judged with the turns before it (context.py — follow-ups, "and for card?", "yes do it"), so one different decision
early on changes what every later turn means. Each WORLD therefore replays the conversation with its OWN earlier
decisions as context — the cascade is the point (the court does the opposite on purpose: it pins earlier turns to
what they should have been, so a mistake is counted once).

Two sets:
    real     multi-turn sessions of real customers (ledger: app + shadow), up to MAX_TURNS turns each — no ground truth,
             so worlds are compared with the live one
    golden   the golden conversations, where every turn has the agent it must reach — right / wrong per world
replay() streams one event per conversation; the finished result is cached for CACHE seconds. Read-only.
"""
from __future__ import annotations

import time

import router
import store

MAX_TURNS = 12
MAX_REAL = 24
CACHE = 600
_cache: dict = {}


def _heads() -> list[str]:
    import constitution
    return constitution.heads_default()[:5]


def _real() -> list[dict]:
    import ledger
    c = ledger._db()
    try:
        ss = [r[0] for r in c.execute("select session from decisions where source in ('app','shadow') and session is not null and text is not null "
                                      "group by session having count(*) >= 2 order by max(at) desc limit ?", (MAX_REAL,))]
        out = []
        for s in ss:
            rows = [dict(r) for r in c.execute("select id, at, text, chosen, by from decisions where session = ? and text is not null order by at", (s,))]
            turns, last = [], None
            for r in rows:                                  # the app logs some turns twice (shadow + app): keep one
                if last and r["text"] == last["text"] and r["at"] - last["at"] < 30:
                    continue
                turns.append({"text": r["text"], "live_then": r["chosen"]})
                last = r
            if len(turns) >= 2:
                out.append({"id": s, "source": "real", "title": f"{len(turns)} turns", "turns": turns[:MAX_TURNS], "at": rows[0]["at"]})
        return out
    finally:
        c.close()


def _golden() -> list[dict]:
    import gates
    return [{"id": t["id"], "source": "golden", "title": t["title"], "lang": t.get("lang"), "turns": [{"text": x["text"], "want": x["want"]} for x in t["turns"]]}
            for t in gates.threads()]


def _play(conv: dict, head, g: dict) -> list[dict]:
    """One world: every turn routed with THIS world's earlier decisions as the conversation so far."""
    thread, out = [], []
    for t in conv["turns"]:
        r = router.route(t["text"], g, head_obj=head, ctx={"thread": thread, "engine": {}})
        chosen = r.get("chosen")
        kind = (router.node(g, chosen or "") or {}).get("kind")
        cx = r.get("context") or {}
        step = {"chosen": chosen, "blocked": r.get("blocked"), "by": r.get("by"), "rule": cx.get("rule"),
                "query": r.get("query") if r.get("query") and r.get("query") != t["text"] else None}
        if "want" in t:
            step["ok"] = chosen == t["want"]
        out.append(step)
        thread.append({"text": t["text"], "standalone": r.get("query") or "", "agent": "engine" if kind == "action" else (chosen or ""),
                       "kind": kind, "answer": "", "at": time.time()})
    return out


def replay(which: str = "both"):
    key = f"{which}:{store.active('router')}:{router.graph().get('updated')}"
    hit = _cache.get(key)
    if hit and time.time() - hit["at"] < CACHE:
        for ev in hit["events"]:
            yield ev
        return
    g = router.graph()
    heads = _heads()
    live = store.active("router")
    names = {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}
    convs = (_real() if which in ("both", "real") else []) + (_golden() if which in ("both", "golden") else [])
    events = [{"type": "start", "heads": heads, "live": live, "names": names, "n": len(convs)}]
    yield events[0]
    loaded = {h: router.load_head(h) for h in heads}
    agg = {h: {"diverge_convs": 0, "diverge_turns": 0, "turns": 0, "golden_ok": 0, "golden_n": 0, "threads_ok": 0, "threads_n": 0} for h in heads}
    for conv in convs:
        worlds = {h: _play(conv, loaded[h], g) for h in heads}
        lv = worlds.get(live) or next(iter(worlds.values()))
        first = None
        for i in range(len(conv["turns"])):
            if len({(w[i]["chosen"], w[i]["blocked"]) for w in worlds.values()}) > 1:
                first = i
                break
        for h, w in worlds.items():
            d = sum(1 for a, b in zip(w, lv) if (a["chosen"], a["blocked"]) != (b["chosen"], b["blocked"]))
            agg[h]["diverge_turns"] += d
            agg[h]["diverge_convs"] += d > 0
            agg[h]["turns"] += len(w)
            if conv["source"] == "golden":
                agg[h]["golden_ok"] += sum(1 for s in w if s.get("ok"))
                agg[h]["golden_n"] += len(w)
                agg[h]["threads_ok"] += all(s.get("ok") for s in w)
                agg[h]["threads_n"] += 1
        ev = {"type": "conversation", **conv, "worlds": worlds, "first_split": first}
        events.append(ev)
        yield ev
    done = {"type": "done", "heads": heads, "live": live, "agg": agg, "n": len(convs)}
    events.append(done)
    _cache.clear()
    _cache[key] = {"at": time.time(), "events": events}
    yield done
