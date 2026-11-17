"""Academy: learn the Studio by doing it — in your own sandbox copy of the flow (staff only).

Each mission is checked from what actually happened, not from a button press where that is possible:
  route     you sent a message through YOUR sandbox flow                  (ledger: source=sandbox, you)
  xray      you opened a decision X-ray                                   (reported by the page)
  build     your sandbox has a new agent connected to the Orchestrator    (your sandbox graph)
  hinglish  a Hinglish message in your sandbox reached the Action agent   (ledger)
  court     you ran a version court and read its gates                    (reported by the page)
Progress is per person; admins see the whole team.
"""
from __future__ import annotations

import time

import ledger
import router
import store

FILE = "academy.json"
MISSIONS = [
    {"id": "route", "title": "Route your first message", "learn": "How a message travels: guardrails → the Orchestrator → an agent.",
     "steps": ["Open Flow and switch on 🧪 Sandbox", "Type any customer message in the console", "Press Send and watch it travel"]},
    {"id": "xray", "title": "Read a decision X-ray", "learn": "Which words drove the decision, and why that agent was chosen.",
     "steps": ["After a message is answered, press 🔬 X-ray", "Find the word that mattered most", "Read the steps from top to bottom"]},
    {"id": "build", "title": "Build an agent", "learn": "Agents are described in plain words; the reasoning model reads the description.",
     "steps": ["In your sandbox press ＋ Agent", "Pick any agent type and write a one-line description", "Make sure the Orchestrator connects to it, then Save"]},
    {"id": "hinglish", "title": "Speak Hinglish to it", "learn": "One Orchestrator, four languages — and how it picks the right reasoning model.",
     "steps": ["In your sandbox console type: mera balance kitna hai", "Send it", "It should reach the Action agent — check the X-ray's language line"]},
    {"id": "court", "title": "Hold a version court", "learn": "Why no model goes live without passing every gate.",
     "steps": ["Open Versions", "Press ⚖ Court & promote on any reflex version", "Read the six gates — you don't have to promote"]},
]


def _all() -> dict:
    return store.read(FILE, {}) or {}


def mark(user: str, mission: str) -> None:
    if mission not in {m["id"] for m in MISSIONS}:
        raise ValueError("no such mission")
    d = _all()
    d.setdefault(user, {}).setdefault(mission, time.time())
    store.write(FILE, d)


def _check(user: str) -> None:
    """Missions verified from evidence."""
    done = _all().get(user, {})
    rows = [r for r in ledger.recent(300, "sandbox") if r.get("session") == user]
    if "route" not in done and rows:
        mark(user, "route")
    if "hinglish" not in done and any(r.get("lang") == "hinglish" and r.get("chosen") == "action" for r in rows):
        mark(user, "hinglish")
    if "build" not in done:
        live = {n["id"] for n in router.graph()["nodes"]}
        sb = router.sandbox_graph(user)
        to = {e["to"] for e in sb["edges"] if e["from"] == "laya" and e.get("enabled", True)}
        if any(n["type"] == "agent" and n["id"] not in live and n["id"] in to and (n.get("description") or "").strip() for n in sb["nodes"]):
            mark(user, "build")


def progress(user: str, admin: bool) -> dict:
    _check(user)
    d = _all()
    mine = d.get(user, {})
    out = {"missions": [dict(m, done=m["id"] in mine, at=mine.get(m["id"])) for m in MISSIONS],
           "done": len(mine), "total": len(MISSIONS)}
    if admin:
        out["team"] = sorted([{"user": u, "done": len(v), "last": max(v.values()) if v else None} for u, v in d.items()],
                             key=lambda x: (-x["done"], -(x["last"] or 0)))
    return out
