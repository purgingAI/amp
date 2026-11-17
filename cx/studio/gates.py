"""Golden sets and the gates a candidate must pass before it may go live.

A golden set is a list of sentences with the agent each one MUST reach — written by people (from
the X-ray, the Teach-me inbox, or by hand), e.g. "where is my refund?" → Action, "how long do UPI
refunds take?" → the refund-policy agent. compare() routes every golden sentence through the full
Orchestrator twice — with the live reflex and with a candidate — and streams the result per
sentence, then the gate matrix:

  no new mistakes      the candidate breaks nothing the live version gets right
  golden accuracy      at least as many right as the live version
  every language       no language (en / hinglish / hi / ar) gets worse
  blind exam           within 1 point of the live version on the unseen blind set
  knows every agent    trained on every agent connected to the Orchestrator
  latency              average routing time not more than 1.5× the live version's (+100 ms)
  consistency          pass^k: money-moving cases must reach the right agent however they are typed (consistency.py)
  regression suite     every case a person graduated to the regression suite still passes (suites.py)
"""
from __future__ import annotations

import json
import os
import time
import uuid

import consistency
import router
import store

GOLDEN = "golden.jsonl"


def golden() -> list[dict]:
    p = os.path.join(store.DATA, GOLDEN)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _write(items: list[dict]) -> None:
    p = store.path(GOLDEN)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def add(text: str, agent: str, by: str, note: str = "", source: str = "manual") -> dict:
    text = " ".join((text or "").split())[:400]
    if len(text) < 3:
        raise ValueError("write the sentence")
    if not router.node(router.graph(), agent):
        raise ValueError("no such agent")
    items = [g for g in golden() if g["text"].lower() != text.lower()]          # one entry per sentence; newest wins
    it = {"id": uuid.uuid4().hex[:10], "text": text, "agent": agent, "lang": router.language(text), "note": note[:200],
          "by": by, "at": time.time(), "source": source}
    items.append(it)
    _write(items)
    return it


def remove(gid: str) -> bool:
    items = golden()
    keep = [g for g in items if g["id"] != gid]
    _write(keep)
    return len(keep) < len(items)


# ── golden conversations: multi-turn threads, each turn judged with the turns before it ──────────
THREADS = "golden_threads.jsonl"


def threads() -> list[dict]:
    f = os.path.join(store.DATA, THREADS)
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()] if os.path.exists(f) else []


def run_threads(head_obj=None, g: dict | None = None) -> dict:
    """Every turn routed with the conversation so far. Earlier turns are recorded as they SHOULD have gone
    (so one mistake is counted once, not again for every turn after it). Turns the engine answers are
    recorded as "engine", exactly as the app path records them."""
    g = g or router.graph()
    out, ok_n, n = [], 0, 0
    for th in threads():
        turns, res = [], []
        for t in th["turns"]:
            if not router.node(g, t["want"]):
                continue
            r = router.route(t["text"], g, head_obj=head_obj, ctx={"thread": turns, "engine": t.get("engine") or {}})
            ok = r.get("chosen") == t["want"]
            cx = r.get("context") or {}
            res.append({"text": t["text"], "want": t["want"], "chosen": r.get("chosen"), "by": r.get("by"), "ok": ok,
                        "rule": cx.get("rule"), "why": cx.get("why"), "query": r.get("query") if r.get("query") != t["text"] else None})
            ok_n += ok
            n += 1
            kind = (router.node(g, t["want"]) or {}).get("kind")
            turns.append({"text": t["text"], "standalone": r.get("query") or "", "agent": "engine" if kind == "action" else t["want"],
                          "kind": kind, "answer": "", "at": time.time()})
        out.append({"id": th["id"], "title": th["title"], "lang": th["lang"], "turns": res, "ok": all(x["ok"] for x in res)})
    return {"threads": out, "right": ok_n, "n": n, "at": time.time()}


def compare(candidate: str, live: str | None = None):
    """Events: {type: item, …} per golden sentence, then {type: summary, gates, …}."""
    g = router.graph()
    live = live or store.active("router")
    heads = {"live": router.load_head(live) if live else None, "cand": router.load_head(candidate)}
    items = [x for x in golden() if router.node(g, x["agent"])]
    res = {"live": [], "cand": []}
    ms = {"live": [], "cand": []}
    trials = {"live": [], "cand": []}                       # per case: [original ok, variation ok, …]
    for i, it in enumerate(items):
        row = {"type": "item", "i": i, "n": len(items), "id": it["id"], "text": it["text"], "lang": it["lang"], "want": it["agent"]}
        for side in ("live", "cand"):
            t0 = time.time()
            r = router.route(it["text"], g, head_obj=heads[side])
            ms[side].append(int((time.time() - t0) * 1000))
            ok = r.get("chosen") == it["agent"]
            res[side].append(ok)
            row[side] = {"chosen": r.get("chosen"), "by": r.get("by"), "ok": ok}
        vs = consistency.variants(it)
        row["variants"] = vs
        for side in ("live", "cand"):
            t = [row[side]["ok"]] + [router.route(v["text"], g, head_obj=heads[side]).get("chosen") == it["agent"] for v in vs]
            trials[side].append(t)
            row[side]["trials"] = t
        row["change"] = "fixed" if row["cand"]["ok"] and not row["live"]["ok"] else "broke" if row["live"]["ok"] and not row["cand"]["ok"] else "same"
        yield row
    n = max(1, len(items))
    langs = sorted({x["lang"] for x in items})
    per_lang = {l: {side: round(100 * sum(ok for ok, x in zip(res[side], items) if x["lang"] == l) /
                                max(1, sum(1 for x in items if x["lang"] == l)), 1) for side in ("live", "cand")} for l in langs}
    vers = {x["id"]: x for x in store.versions()["router"]["list"]}
    blind = {side: (vers.get(v) or {}).get("scores", {}).get("blind") for side, v in (("live", live), ("cand", candidate))}
    # per-agent blind, so a candidate can't quietly erode one agent while its overall number looks fine.
    # Tolerance is 1 whole sentence: a single-sentence wobble on a small agent (e.g. human, 15 sentences) is
    # binomial noise, but a systematic slide (e.g. action 536→520) still trips it.
    def _per(v):
        bpa = (vers.get(v) or {}).get("scores", {}).get("blind_per_agent") or {}
        out = {}
        for a, s in bpa.items():
            try:
                r, t = str(s).split("/"); out[a] = (int(r), int(t))
            except ValueError:
                pass
        return out
    bp_live, bp_cand = _per(live), _per(candidate)
    dropped_agents = [a for a in bp_cand if a in bp_live and bp_cand[a][0] < bp_live[a][0] - 1]
    connected = [c["id"] for c in router.candidates(g)]
    knows = set(heads["cand"]["clf"].classes_) if heads["cand"] else set()
    avg = {side: (sum(ms[side]) / len(ms[side])) if ms[side] else 0 for side in ms}
    fixed = [items[i]["text"] for i in range(len(items)) if res["cand"][i] and not res["live"][i]]
    broke = [items[i]["text"] for i in range(len(items)) if res["live"][i] and not res["cand"][i]]
    if threads():
        yield {"type": "phase", "text": f"Replaying {len(threads())} golden conversations, turn by turn, with both versions…"}
    conv = {side: run_threads(heads[side], g) for side in ("live", "cand")} if threads() else None
    cons = consistency.gate(items, trials, g, router.node)
    import suites
    suite = suites.gate(items, res)
    try:
        import constitution
        law = constitution.violations(heads["cand"], heads["live"], g) if constitution.ratified() else None
    except Exception:                                       # noqa: BLE001 — the court still sits without it
        law = None
    gates = [
        {"name": "No new mistakes", "pass": not broke, "detail": f"{len(broke)} golden sentence(s) the live version gets right would break" if broke else "nothing the live version gets right breaks"},
        {"name": "Golden accuracy", "pass": sum(res["cand"]) >= sum(res["live"]), "detail": f"{sum(res['cand'])}/{len(items)} vs live {sum(res['live'])}/{len(items)}"},
        {"name": "Every language", "pass": all(v["cand"] >= v["live"] for v in per_lang.values()),
         "detail": ", ".join(f"{l} {v['cand']:g}% (live {v['live']:g}%)" for l, v in per_lang.items()) or "no golden sentences"},
        {"name": "Blind exam", "pass": blind["cand"] is None or blind["live"] is None or blind["cand"] >= blind["live"] - 0.01,
         "detail": f"{(blind['cand'] or 0) * 100:.1f}% vs live {(blind['live'] or 0) * 100:.1f}% (unseen sentences)" if blind["cand"] is not None else "not measured"},
        {"name": "Every agent", "pass": not dropped_agents,
         "detail": ("no agent drops more than noise on the blind set" if not dropped_agents else
                    "drops: " + ", ".join(f"{router.node(g, a)['name'] if router.node(g, a) else a} {bp_live[a][0]}→{bp_cand[a][0]}/{bp_cand[a][1]}" for a in dropped_agents)) if (bp_live and bp_cand) else "per-agent blind not measured"},
        {"name": "Knows every agent", "pass": all(a in knows for a in connected),
         "detail": "trained on all connected agents" if all(a in knows for a in connected) else "has not learned: " + ", ".join(router.node(g, a)["name"] for a in connected if a not in knows)},
        {"name": "Conversations", "pass": conv is None or conv["cand"]["right"] >= conv["live"]["right"],
         "detail": f"{conv['cand']['right']}/{conv['cand']['n']} follow-up turns vs live {conv['live']['right']}/{conv['live']['n']}" if conv else "no golden conversations"},
        {"name": "Constitution", "pass": law is None or not law["broken"],
         "detail": "no ratified articles yet" if law is None else
                   (f"keeps {law['cand']}/{law['n']} articles (live {law['live']}/{law['n']})" if not law["broken"] else "breaks " + ", ".join(law["broken"]) + " — which the live version keeps")},
        {"name": "Latency", "pass": avg["cand"] <= avg["live"] * 1.5 + 100, "detail": f"{avg['cand']:.0f} ms vs live {avg['live']:.0f} ms per message"},
        {"name": f"Consistency (pass^{cons['k']})", "pass": cons["pass"], "detail": cons["detail"]},
        {"name": "Regression suite", "pass": suite["pass"], "detail": suite["detail"]},
    ]
    yield {"type": "summary", "candidate": candidate, "live": live, "n": len(items), "right": {k: sum(v) for k, v in res.items()},
           "fixed": fixed, "broke": broke, "per_lang": per_lang, "blind": blind, "ms": avg, "gates": gates, "consistency": cons, "suites": suite,
           "pass": all(x["pass"] for x in gates)}
