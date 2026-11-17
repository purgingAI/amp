"""Flow versions: every meaningful save of the live graph is kept, with who saved it and what changed.

A version is a full copy of the graph (small JSON) plus a diff against the previous one:
agents added / removed / changed (name, description, on/off, knowledge base, examples, webhook),
wires added / removed / paused, and Orchestrator settings. Moving cards around is not a version.
Restoring an old version saves it as a NEW version — history is never rewritten.
"""
from __future__ import annotations

import json
import os
import time

import store

DIR = os.path.join(store.DATA, "flow_versions")
WATCH = ("name", "description", "enabled", "collection", "url", "method", "body", "answer_path", "answer_template", "min_coverage", "kind")


def _edges(g: dict) -> dict[tuple, dict]:
    return {(e["from"], e["to"]): e for e in g.get("edges", [])}


def diff(a: dict | None, b: dict) -> dict:
    a = a or {"nodes": [], "edges": [], "settings": {}}
    na, nb = {n["id"]: n for n in a.get("nodes", [])}, {n["id"]: n for n in b.get("nodes", [])}
    ea, eb = _edges(a), _edges(b)
    changed = []
    for i in set(na) & set(nb):
        f = [k for k in WATCH if na[i].get(k) != nb[i].get(k)]
        if sorted(na[i].get("examples") or []) != sorted(nb[i].get("examples") or []):
            f.append(f"examples ({len(na[i].get('examples') or [])}→{len(nb[i].get('examples') or [])})")
        if f:
            changed.append({"id": i, "name": nb[i].get("name", i), "fields": f})
    name = lambda g, i: (g.get(i) or {}).get("name", i)  # noqa: E731
    out = {
        "nodes_added": [{"id": i, "name": nb[i].get("name", i), "type": nb[i].get("type")} for i in nb if i not in na],
        "nodes_removed": [{"id": i, "name": na[i].get("name", i)} for i in na if i not in nb],
        "nodes_changed": changed,
        "edges_added": [f"{name(nb, f)} → {name(nb, t)}" for (f, t) in eb if (f, t) not in ea],
        "edges_removed": [f"{name(na, f)} → {name(na, t)}" for (f, t) in ea if (f, t) not in eb],
        "edges_toggled": [f"{name(nb, f)} → {name(nb, t)}: {'paused' if eb[(f, t)].get('enabled') is False else 'resumed'}"
                          for (f, t) in eb if (f, t) in ea and (ea[(f, t)].get("enabled", True) != eb[(f, t)].get("enabled", True))],
        "settings": {k: [a.get("settings", {}).get(k), v] for k, v in b.get("settings", {}).items() if a.get("settings", {}).get(k) != v},
    }
    out["empty"] = not any(out[k] for k in out)
    return out


def _all() -> list[int]:
    if not os.path.isdir(DIR):
        return []
    return sorted(int(f[:-5]) for f in os.listdir(DIR) if f.endswith(".json") and f[:-5].isdigit())


def latest() -> dict | None:
    ns = _all()
    return json.load(open(os.path.join(DIR, f"{ns[-1]}.json"), encoding="utf-8")) if ns else None


def snapshot(g: dict, by: str, note: str = "") -> dict | None:
    prev = latest()
    d = diff(prev["graph"] if prev else None, g)
    if prev and d["empty"]:
        return None                                         # only cards moved — not a new version
    os.makedirs(DIR, exist_ok=True)
    n = (prev["n"] + 1) if prev else 1
    rec = {"n": n, "at": time.time(), "by": by, "note": note[:200], "diff": d, "graph": g}
    store.write(f"flow_versions/{n}.json", rec)
    return {k: v for k, v in rec.items() if k != "graph"}


def listing(limit: int = 40) -> list[dict]:
    out = []
    for n in reversed(_all()[-limit:]):
        r = json.load(open(os.path.join(DIR, f"{n}.json"), encoding="utf-8"))
        out.append({k: v for k, v in r.items() if k != "graph"})
    return out


def get(n: int) -> dict | None:
    p = os.path.join(DIR, f"{n}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
