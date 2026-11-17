"""The Time Machine — counterfactual repair of one past decision.

Pick a decision the Orchestrator got wrong and say where it SHOULD have gone. The Time Machine rewinds to
that moment and searches for the smallest change that would have sent it to the right agent — a nudge to the
reflex bar, the reasoning thresholds, the fallback, or (durably) teaching it — and for each one it REPLAYS the
whole golden set and every golden conversation to see what else that change would move. A fix is "safe" only
when it flips this one and breaks nothing else; the rest are shown struck-through so you can see the cost.

Nothing is applied automatically. diagnose() streams the search move by move (so the UI can show the hunt as
it happens); apply() routes an approved fix to the same paths the rest of the Studio already uses — a bar
change saves a flow version, teaching adds a golden sentence and a training label.

    reuses  ledger (the misroute) · router.route (replay against a modified graph) · gates (golden + threads)
            · flowver (bar change = a flow version) · inbox/gates.add (teach + lock in)
"""
from __future__ import annotations

import copy
import time

import gates
import ledger
import router
import store

# the settings we can nudge, and the range we scan each over
BARS = [round(x / 100, 2) for x in range(25, 81, 5)]           # reflex / reasoning thresholds 0.25 … 0.80


def _name(g: dict, aid: str | None) -> str:
    n = router.node(g, aid or "")
    return (n or {}).get("name") or aid or "—"


def _route(text: str, g: dict) -> dict:
    return router.route(text, g)


def _baseline(g: dict) -> dict:
    """What the live graph gets right on every golden sentence — the yardstick for 'what else moved'. The
    golden-conversation replay is costly (a Claude rewrite per follow-up), so it is computed lazily, only when
    a candidate is otherwise clean and we need to confirm it."""
    items = [x for x in gates.golden() if router.node(g, x["agent"])]
    gold = {it["id"]: (_route(it["text"], g).get("chosen") == it["agent"]) for it in items}
    return {"g": g, "items": items, "gold": gold, "gold_right": sum(gold.values()), "thread_right": None, "thread_n": None}


def _baseline_threads(base: dict) -> None:
    if base["thread_right"] is None:
        th = gates.run_threads(g=base["g"])
        base["thread_right"], base["thread_n"] = th["right"], th["n"]


def _blast(g2: dict, base: dict) -> dict:
    """Replay the golden set on the modified graph; what breaks that the live graph gets right. Only when the
    golden set is clean do we also replay the golden conversations (the expensive check)."""
    broke, fixed, right = [], 0, 0
    for it in base["items"]:
        ok = _route(it["text"], g2).get("chosen") == it["agent"]
        right += ok
        was = base["gold"].get(it["id"])
        if was and not ok:
            broke.append({"text": it["text"], "want": _name(g2, it["agent"])})
        elif ok and not was:
            fixed += 1
    out = {"gold_right": right, "gold_was": base["gold_right"], "gold_total": len(base["items"]),
           "broke": broke, "fixed": fixed, "thread_right": None, "thread_was": None, "thread_n": None, "thread_regressions": 0}
    if broke:                                                # already unsafe on golden — no need to pay the thread cost
        out["regressions"] = len(broke)
        return out
    _baseline_threads(base)
    th = gates.run_threads(g=g2)
    out.update(thread_right=th["right"], thread_was=base["thread_right"], thread_n=th["n"],
               thread_regressions=max(0, base["thread_right"] - th["right"]))
    out["regressions"] = max(0, base["thread_right"] - th["right"])
    return out


def _with(g: dict, **settings) -> dict:
    g2 = copy.deepcopy(g)
    g2.setdefault("settings", {}).update(settings)
    return g2


def _flip_scan(text: str, g: dict, want: str, key: str) -> tuple[float, dict] | None:
    """The setting value CLOSEST to the current one that flips this message to `want`. Values are tried
    nearest-first and the first flip wins, so the answer is the smallest change — and we stop early."""
    cur = g["settings"].get(key, 0.45 if key == "reflex_threshold" else 0.5)
    for v in sorted((x for x in BARS if abs(x - cur) > 1e-9), key=lambda x: (abs(x - cur), x)):
        if _route(text, _with(g, **{key: v})).get("chosen") == want:
            return v, {"from": cur, "to": v}
    return None


def diagnose(did: int, want: str):
    """Stream the rewind + the search. Events: moment, probe, result, durable, done."""
    row = ledger.get(int(did))
    if not row:
        yield {"type": "error", "error": "no such decision"}
        return
    g = router.graph()
    if not router.node(g, want):
        yield {"type": "error", "error": "pick the agent it should have reached"}
        return
    text = row["text"]
    now = _route(text, g)
    got = now.get("chosen") or now.get("blocked")
    rx = now.get("reflex") or {}
    conf = rx.get("confidence")
    checks = len([x for x in gates.golden() if router.node(g, x["agent"])])
    yield {"type": "moment", "ledger": row["id"], "text": text, "at": row.get("at"), "got": got, "got_name": _name(g, got),
           "by": now.get("by"), "conf": round(conf, 3) if conf is not None else None, "want": want, "want_name": _name(g, want),
           "checks": checks, "threads": len(gates.threads()), "already": got == want}
    if got == want:
        yield {"type": "done", "safe": 0, "best": None, "note": "it already routes there"}
        return

    base = None                                              # golden/thread yardstick, built lazily on the first flip

    # the reflex bar always matters; the reasoning / new-agent bars only matter when the reasoning model was
    # actually consulted (route() records "laya" only then — a confident reflex never asks it, so changing its
    # thresholds could not have mattered)
    levers = [("reflex_threshold", "Reflex bar", "how sure the fast brain must be")]
    if now.get("laya"):
        levers.append(("laya_threshold", "Reasoning bar", "how sure the reasoning model must be"))
        if want in (now.get("reflex") or {}).get("untrained", []):
            levers.append(("new_agent_threshold", "New-agent bar", "the bar for an agent the reflex has never seen"))

    safe, best = 0, None
    for key, title, how in levers:
        yield {"type": "probe", "id": key, "lever": title, "testing": how}
        scan = _flip_scan(text, g, want, key)
        if not scan:
            yield {"type": "result", "id": key, "lever": title, "flips": False, "safe": False, "note": "no value flips this one"}
            continue
        to, change = scan
        if base is None:
            base = _baseline(g)
        bl = _blast(_with(g, **{key: to}), base)
        is_safe = bl["regressions"] == 0
        if is_safe:
            safe += 1
            best = best or key
        yield {"type": "result", "id": key, "lever": title, "flips": True, "safe": is_safe, "fix": {"kind": "setting", "setting": key, "to": to},
               "change": {"label": f"{change['from']:.2f} → {change['to']:.2f}", **change}, "blast": bl}

    # teaching it: durable, never breaks routing today, feeds the next challenger + guards every court
    yield {"type": "probe", "id": "teach", "lever": "Teach it", "testing": "lock the right answer in"}
    yield {"type": "durable", "id": "teach", "lever": "Teach it", "want": want, "want_name": _name(g, want), "fix": {"kind": "teach"},
           "note": "adds a golden sentence (every future version is checked against it) and a training label for the next challenger — the durable fix when no setting flips it cleanly"}

    yield {"type": "done", "safe": safe, "best": best}


def apply(did: int, want: str, fix: dict, by: str) -> dict:
    """Apply an approved fix through the Studio's existing paths. Admin only (enforced by app.py)."""
    g = router.graph()
    row = ledger.get(int(did))
    if not router.node(g, want):
        raise ValueError("no such agent")
    kind = fix.get("kind")
    if kind == "setting":
        key, to = fix.get("setting"), float(fix.get("to"))
        if key not in ("reflex_threshold", "laya_threshold", "new_agent_threshold") or not 0.2 <= to <= 0.9:
            raise ValueError("bad setting")
        g2 = _with(g, **{key: to})
        saved = router.save_graph(g2, by)                       # saves a flow version (flowver)
        return {"ok": True, "applied": key, "to": to, "flow": (saved.get("_version") or {}).get("n")}
    if kind == "teach":
        text = (row or {}).get("text") or fix.get("text") or ""
        if row:
            ledger.set_label(row["id"], want, by)
        item = gates.add(text, want, by, note="locked in from the Time Machine", source=f"timemachine:{did}")
        try:
            import inbox
            inbox.label_text(text, want, by)                    # feeds the next challenger
        except Exception:                                       # noqa: BLE001
            pass
        return {"ok": True, "applied": "teach", "golden": item}
    raise ValueError("unknown fix")
