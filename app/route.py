"""Expert routing — 'who should handle this?' — the ACE brain for who-to-call.

Combines the CODE-understanding graph (who actually owns/wrote the modules that implement a topic)
with the people graph (tech expertise + what they're working on now) and, optionally, the trust
layer (rapport with the requester). Code ownership is weighted highest — the person whose code
implements the thing is usually the right call.
"""
from __future__ import annotations
from . import codegraph, knowledge
try:
    from . import understand
except Exception:
    understand = None


def route_expert(topic: str, requester: str | None = None, limit: int = 3) -> dict:
    topic = (topic or "").strip()
    if not topic:
        return {"topic": topic, "experts": [], "modules": []}

    owner_score: dict = {}
    owns_module: dict = {}
    owns_commits: dict = {}
    modules: list = []
    if codegraph.enabled():
        mods = codegraph.search_capabilities(topic, limit=4)
        n = len(mods)
        for i, m in enumerate(mods):
            modules.append({"module": m["module"], "purpose": m.get("purpose")})
            owners = m.get("owners", [])
            total = sum(int(o.get("commits", 0) or 0) for o in owners) or 1
            mod_weight = (n - i)   # the most-relevant module's owners count most
            for o in owners:
                commits = int(o.get("commits", 0) or 0)
                share = commits / total                      # owner's share of THIS module (0..1)
                owner_score[o["name"]] = owner_score.get(o["name"], 0) + mod_weight * share * 50
                # remember the module where this person owns the biggest share
                if share > owns_commits.get(o["name"], (0, ""))[0]:
                    owns_commits[o["name"]] = (share, m["module"])
                    owns_module[o["name"]] = m["module"]

    exp_score: dict = {}
    focus: dict = {}
    if knowledge.enabled():
        for e in knowledge.find_experts(topic, limit=6):
            exp_score[e["name"]] = e.get("score", 0)
            focus[e["name"]] = e.get("current")

    names = set(owner_score) | set(exp_score)
    ranked = []
    for n in names:
        # weighted code-ownership (already rank/share-scaled) + a bounded expertise contribution
        score = owner_score.get(n, 0) + min(exp_score.get(n, 0), 120)
        share = owns_commits.get(n, (0, ""))[0]
        rec = {"name": n, "score": round(score, 1), "owns_module": owns_module.get(n),
               "ownership_pct": round(share * 100) if share else None, "current_focus": focus.get(n)}
        if requester and understand is not None:
            rel = understand.relationship(requester, n)
            rec["trust_with_you"] = rel.get("trust")
            rec["connected"] = rel.get("connected")
        ranked.append(rec)
    ranked.sort(key=lambda x: (-(x["score"]), -(x.get("trust_with_you") or 0)))

    top = ranked[:limit]
    for r in top:
        why = []
        if r.get("owns_module"):
            pct = f" ({r['ownership_pct']}% of it)" if r.get("ownership_pct") else ""
            why.append(f"owns {r['owns_module']}{pct}")
        if r.get("current_focus"):
            why.append(f"currently on {r['current_focus']}")
        if r.get("trust_with_you"):
            why.append(f"trust {r['trust_with_you']} with you")
        r["why"] = "; ".join(why) or "worked in this area"
    return {"topic": topic, "experts": top, "modules": modules[:3]}
