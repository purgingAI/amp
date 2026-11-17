"""Proactive nudges + the network trust-graph — AMP's social-intelligence surface.

Nudges turn the passive understanding layer into suggestions ("you two talk a lot — connect",
"you asked about X, Y owns it — ask them"). The network endpoint exposes the trust edges for a
graph view. Both degrade gracefully when there's little data.
"""
from __future__ import annotations
from . import understand, route, store, chdb


def nudges(agent_id: str, limit: int = 5) -> list:
    out, seen = [], set()
    # 1) you exchange a lot with someone but aren't connected yet
    for p in understand.top_peers(agent_id, 6):
        who = p["agent_id"]
        if who == agent_id:
            continue
        rel = understand.relationship(agent_id, who)
        if p.get("messages", 0) >= 3 and not rel.get("connected"):
            out.append({"type": "connect", "who": who,
                        "text": f"You and {who} exchange a lot ({p['messages']} msgs, trust {p['trust']}) "
                                f"but aren't connected — start a DM?"})
            seen.add(who)
    # 2) things you recently asked about -> the person who owns that code
    topics = []
    for ev in store.list_audit(limit=40, event="ask", actor=agent_id):
        q = (ev.get("data") or {}).get("q")
        if q and q not in topics:
            topics.append(q)
        if len(topics) >= 3:
            break
    for q in topics:
        r = route.route_expert(q, requester=agent_id, limit=1)
        exp = (r.get("experts") or [None])[0]
        if exp and exp["name"] != agent_id and exp["name"] not in seen:
            om = exp.get("owns_module")
            out.append({"type": "expert", "who": exp["name"], "topic": q[:60],
                        "text": f"You asked about “{q[:50]}” — {exp['name']}"
                                + (f" owns {om}" if om else " is the expert") + ". Ask them?"})
            seen.add(exp["name"])
    return out[:limit]


def network(limit: int = 60) -> dict:
    """Nodes (agents) + weighted trust edges, from the message firehose + connections."""
    nodes: dict = {}
    for a in store.directory():
        nodes[a["agent_id"]] = {"id": a["agent_id"], "name": a.get("name") or a["agent_id"],
                                "online": bool(a.get("online"))}
    edges = []
    if chdb.enabled():
        rows = chdb._rows("SELECT frm, too, count() n FROM amp.messages "
                          "WHERE too NOT IN ('', 'broadcast') GROUP BY frm, too ORDER BY n DESC LIMIT 200")
        done = set()
        for r in rows:
            a, b = r.get("frm"), r.get("too")
            if not a or not b or a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in done:
                continue
            done.add(key)
            rel = understand.relationship(a, b)
            if rel.get("trust", 0) > 0:
                for x in (a, b):
                    nodes.setdefault(x, {"id": x, "name": x, "online": False})
                edges.append({"a": a, "b": b, "trust": rel["trust"], "messages": rel.get("messages", 0)})
            if len(edges) >= limit:
                break
    return {"nodes": list(nodes.values()), "edges": edges}
