"""Two suites inside the golden set: what must never break, and what we are still climbing towards.

  regression   cases the router has mastered. A candidate must keep every one (court gate "Regression suite").
  capability   cases it is still learning. They are reported in the court and may climb; they never block alone.

A capability case that has held on the last GENERATIONS live versions, in every variation (pass^k, consistency.py),
is PROPOSED for graduation — a person accepts it; nothing moves itself. A case that no live generation ever got right
as written is flagged SUSPECT: the label may be wrong, or the case may ask for something no agent offers.
(Anthropic, "Demystifying evals for AI agents", Jan 2026: keep regression near 100%, let capability evals climb,
and treat a task everything fails as a possibly broken task before blaming the agent.)

Results are computed once per (golden set, lineage, flow) and cached; the court never waits for them.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import consistency
import gates
import router
import store

GENERATIONS = 3                                             # mastery = held on the last 3 live versions
SHOWN = 5                                                   # how many live generations the history dots show
_cache: dict = {}


def lineage(n: int = SHOWN) -> list[str]:
    """Distinct versions in the order they went live, oldest first, ending with the live one."""
    v = store.versions()["router"]
    seen = []
    for h in v.get("history") or []:
        if h["id"] in seen:
            seen.remove(h["id"])
        seen.append(h["id"])
    if v.get("active") and (not seen or seen[-1] != v["active"]):
        seen.append(v["active"])
    return seen[-n:]


def role(item: dict) -> str:
    return item.get("role") if item.get("role") in ("regression", "capability") else "capability"


def _key(items, gens, g) -> str:
    gp = os.path.join(store.DATA, gates.GOLDEN)
    raw = json.dumps([[i["id"], i["text"], i["agent"]] for i in items] + gens + [json.dumps(g.get("edges"), sort_keys=True), consistency.VERSION])
    return hashlib.sha256(raw.encode()).hexdigest()[:16] + str(int(os.path.getmtime(gp)) if os.path.exists(gp) else 0)


def compute(fresh: bool = False):
    """Events: {type: version, id, i, n} while scoring each live generation, then {type: suites, …}."""
    g = router.graph()
    items = [x for x in gates.golden() if router.node(g, x["agent"])]
    gens = lineage()
    k = _key(items, gens, g)
    if not fresh and k in _cache:
        yield _cache[k]
        return
    hist = {it["id"]: {} for it in items}
    for vi, v in enumerate(gens):
        yield {"type": "version", "id": v, "i": vi, "n": len(gens)}
        try:
            head = router.load_head(v)
        except Exception:                                   # noqa: BLE001 — a pruned head simply has no dot
            continue
        for it in items:
            t = [router.route(it["text"], g, head_obj=head).get("chosen") == it["agent"]]
            t += [router.route(x["text"], g, head_obj=head).get("chosen") == it["agent"] for x in consistency.variants(it)]
            hist[it["id"]][v] = {"ok": t[0], "trials": t, "consistent": all(t)}
    recent = gens[-GENERATIONS:]
    out = []
    for it in items:
        h = hist[it["id"]]
        r = role(it)
        mastered = len(recent) == GENERATIONS and all((h.get(v) or {}).get("consistent") for v in recent)
        never = bool(h) and not any(x["ok"] for x in h.values())
        live = h.get(gens[-1]) if gens else None
        status = ("regressed" if r == "regression" and live and not live["ok"] else
                  "graduate" if r == "capability" and mastered else
                  "suspect" if never else r)
        out.append({"id": it["id"], "text": it["text"], "lang": it["lang"], "agent": it["agent"], "role": r, "status": status,
                    "history": [{"v": v, **(h.get(v) or {"missing": True})} for v in gens],
                    "role_by": it.get("role_by"), "role_at": it.get("role_at")})
    res = {"type": "suites", "generations": gens, "mastery": recent, "k": consistency.K, "items": out, "at": time.time(),
           "counts": {s: sum(1 for x in out if x["status"] == s) for s in ("regression", "capability", "graduate", "suspect", "regressed")}}
    _cache.clear()
    _cache[k] = res
    yield res


def set_role(gid: str, new: str, by: str) -> dict:
    if new not in ("regression", "capability"):
        raise ValueError("role is regression or capability")
    items = gates.golden()
    hit = next((x for x in items if x["id"] == gid), None)
    if not hit:
        raise ValueError("no such golden sentence")
    hit.update(role=new, role_by=by, role_at=time.time())
    gates._write(items)
    return hit


def gate(items: list[dict], res: dict) -> dict:
    """Court gate: the candidate keeps EVERY regression case (as written). Capability is reported, not gated."""
    reg = [i for i, it in enumerate(items) if role(it) == "regression"]
    cap = [i for i, it in enumerate(items) if role(it) != "regression"]
    lost = [items[i]["text"] for i in reg if not res["cand"][i]]
    cnt = {side: {"regression": sum(res[side][i] for i in reg), "capability": sum(res[side][i] for i in cap)} for side in res}
    return {"n": {"regression": len(reg), "capability": len(cap)}, "right": cnt, "lost": lost, "pass": not lost,
            "detail": ("no regression suite yet — graduate mastered cases in Golden sets" if not reg else
                       f"keeps {cnt['cand']['regression']}/{len(reg)} regression cases"
                       + (f" — loses: {', '.join(lost[:3])}" if lost else "")
                       + f" · capability {cnt['cand']['capability']}/{len(cap)} (live {cnt['live']['capability']})")}
