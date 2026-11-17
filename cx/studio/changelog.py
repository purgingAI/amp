"""Change log: one timeline of everything that changed how customers are answered — who, when, what.

Built from the records each part already keeps, so nothing can be forgotten by a code path that
did not remember to log:
  versions.json            reflex promotions                    (router history)
  collections/*/versions   retriever promotions per knowledge base
  action_history.json      intent-model (action agent) switches
  flow_versions/           every saved flow, with its diff
  learning.json audit      rollout, shadow, nightly switches and targets
  calibration.json audit   calibrated confidence on / off
  rollout_guard.json       automatic rollout switch-offs, with the reason
"""
from __future__ import annotations

import json
import os

import store


def _name(by: str | None) -> str:
    return (by or "system").split("@")[0]


def timeline(limit: int = 120) -> list[dict]:
    ev: list[dict] = []
    v = store.versions()
    for h in v.get("router", {}).get("history", []):
        ev.append({"at": h["at"], "by": _name(h.get("by")), "kind": "promote", "title": f"Reflex {h['id']} promoted",
                   "detail": "the Orchestrator's fast router"})
    cdir = os.path.join(store.DATA, "collections")
    for cid in (os.listdir(cdir) if os.path.isdir(cdir) else []):
        kv = store.read(f"collections/{cid}/versions.json") or {}
        for h in kv.get("history", []):
            ev.append({"at": h["at"], "by": _name(h.get("by")), "kind": "promote", "title": f"Retriever {h['id']} promoted",
                       "detail": f"knowledge base {cid}"})
    for h in store.read("action_history.json", []) or []:
        ev.append({"at": h["at"], "by": _name(h.get("by")), "kind": "promote", "title": f"Intent model {h.get('to')} live",
                   "detail": f"action agent, replacing {h.get('from') or '—'}"})
    try:
        import flowver
        for f in flowver.listing(60):
            d = f.get("diff") or {}
            bits = []
            for k, lab in (("nodes_added", "added"), ("nodes_removed", "removed")):
                if d.get(k):
                    bits.append(f"{lab} " + ", ".join(n.get("name", "") for n in d[k][:3]))
            if d.get("nodes_changed"):
                bits.append("changed " + ", ".join(n.get("name", "") for n in d["nodes_changed"][:3]))
            if d.get("edges_added") or d.get("edges_removed"):
                bits.append(f"{len(d.get('edges_added', []))} wire(s) added, {len(d.get('edges_removed', []))} removed")
            for k, (a, b) in (d.get("settings") or {}).items():
                if a is not None:
                    bits.append(f"{k.replace('_', ' ')} {a} → {b}")
            ev.append({"at": f["at"], "by": _name(f.get("by")), "kind": "flow", "title": f"Flow v{f['n']} saved",
                       "detail": "starting point" if f["n"] == 1 else ("; ".join(bits) or "no visible change"),
                       "note": f.get("note") or ""})
    except Exception:                                       # noqa: BLE001
        pass
    for a in (store.read("learning.json") or {}).get("audit", []):
        for system, ch in (a.get("change") or {}).items():
            for k, val in ch.items():
                if system == "rollout" and k == "percent":
                    title, kind = (f"Customer-app rollout → {val}%", "rollout")
                elif k in ("enabled", "nightly"):
                    title, kind = (f"{'Shadow' if system == 'shadow' else system.title()} {k} {'on' if val else 'off'}", "switch")
                else:
                    title, kind = (f"{system.title()} {k.replace('_', ' ')} → {val}", "switch")
                ev.append({"at": a["at"], "by": _name(a.get("by")), "kind": "guard" if a.get("by") == "rollout-guard" else kind,
                           "title": title, "detail": ""})
    for a in (store.read("calibration.json") or {}).get("audit", []):
        ev.append({"at": a["at"], "by": _name(a.get("by")), "kind": "switch",
                   "title": f"Calibrated confidence {'on' if a.get('applied') else 'off'}", "detail": ""})
    for t in (store.read("rollout_guard.json") or {}).get("events", []):
        ev.append({"at": t["at"], "by": "rollout-guard", "kind": "guard", "title": f"Rollout switched OFF automatically (was {t['from_percent']}%)",
                   "detail": t.get("why", "")})
    # the guard's own learning.json audit line duplicates its event; keep the one with the reason
    seen, out = set(), []
    for e in sorted(ev, key=lambda e: -e["at"]):
        key = (round(e["at"]), e["kind"]) if e["kind"] == "guard" else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(e)
    return out[:limit]
